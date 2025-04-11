import os
import uuid
import requests
import base64
from typing import Optional
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from agent_engine.agent import BaseChatEngine


class ContextualChatEngine(BaseChatEngine):
    def __init__(
        self,
        model_name: str,
        max_pixels: int = 660 * 660,
        min_pixels: int = 128 * 128,
        system_prompt: Optional[str] = None,
        tmp_dir: str = "asset/tmp",
        max_new_tokens: int = 512
    ):
        super().__init__(model_name, max_pixels, min_pixels, system_prompt, tmp_dir)
        self.is_online = model_name.startswith('http')

        if self.is_online:
            parts = model_name.split('@', 2)
            if len(parts) != 3:
                raise ValueError("[AGENT] Invalid model name for online mode.")
            self.api_url, self.api_key, self.api_model = parts
            print(
                f"[AGENT] Online mode with API URL: {self.api_url}")
            print(
                f"[AGENT] Initializing online mode with model: {self.api_model}")
            self.model = None
            self.processor = None
        else:
            print(
                f"[AGENT] Initializing local model: {model_name}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name, use_fast=True)

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        self.tmp_dir = tmp_dir
        self.max_new_tokens = max_new_tokens
        self.tasks = {}
        self.context = []
        if system_prompt:
            self._init_system_prompt(system_prompt)

    def _init_system_prompt(self, system_prompt: str):
        """Initialize the system prompt."""
        self.system_prompt = {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        }
        self.context.append(self.system_prompt)

    def generate_response(self, prompt: str, img_path: Optional[str] = None) -> dict:
        job_id = str(uuid.uuid4())

        if self.is_online:
            content = []

            if img_path:
                if not os.path.exists(img_path):
                    return {
                        'status': 'error',
                        'result': f"Image file not found: {img_path}",
                        'prompt': prompt,
                        'image_path': img_path
                    }

                try:
                    with open(img_path, "rb") as image_file:
                        encoded_string = base64.b64encode(
                            image_file.read()).decode()
                    image_url = f"data:image/jpeg;base64,{encoded_string}"

                    # 添加图片内容部分
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "auto"
                        }
                    })
                except Exception as e:
                    return {
                        'status': 'error',
                        'result': f"Failed to process image: {str(e)}",
                        'prompt': prompt,
                        'image_path': img_path
                    }

            # 添加文本内容部分
            content.append({
                "type": "text",
                "text": prompt
            })

            # 构造完整消息
            messages = self.context.copy()
            messages.append({
                "role": "user",
                "content": content
            })

            # 发送API请求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.api_model,
                "messages": messages,
                "max_tokens": self.max_new_tokens,
                "temperature": 0.7,
                "response_format": {"type": "text"},
                "top_p": 0.7,
                "n": 1,
                "stream": False
            }

            try:
                response = requests.post(
                    self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

                # 处理响应并更新上下文
                output_text = result['choices'][0]['message']['content']
                self.context.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": output_text}]
                })

                return {
                    'status': 'completed',
                    'result': output_text,
                    'prompt': prompt,
                    'image_path': img_path
                }
            except Exception as e:
                return {
                    'status': 'error',
                    'result': str(e),
                    'prompt': prompt,
                    'image_path': img_path
                }
        else:
            # 本地模型处理
            if img_path:
                img = Image.open(img_path).convert("RGB")
                img_path = os.path.join(self.tmp_dir, f"{job_id}.jpg")
                img.save(img_path)

            self.tasks[job_id] = {
                'status': 'pending',
                'result': None,
                'prompt': prompt,
                'image_path': img_path
            }

            return self._process_task(job_id)

    def _process_task(self, job_id: str) -> dict:
        """Process a specific task with the given job ID."""
        try:
            task = self.tasks[job_id]
            img_path = task['image_path']
            prompt = task['prompt']

            messages = self.context.copy()
            user_content = []

            if img_path:
                user_content.append({
                    "type": "image",
                    "image": img_path,
                    "max_pixels": self.max_pixels,
                    "min_pixels": self.min_pixels
                })

            user_content.append({
                "type": "text",
                "text": prompt
            })

            user_msg = {"role": "user", "content": user_content}
            messages.append(user_msg)

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            self.tasks[job_id]['status'] = 'completed'
            self.tasks[job_id]['result'] = output_text
            self.context.append({"role": "assistant", "content": [
                                {"type": "text", "text": output_text}]})

            return self.tasks[job_id]

        except Exception as e:
            self.tasks[job_id]['status'] = 'error'
            self.tasks[job_id]['result'] = str(e)
            print(f"Error processing task {job_id}: {e}")
            if img_path and os.path.exists(img_path):
                os.remove(img_path)

    def clear_context(self):
        """Clear the context and clean up temporary files."""
        self.context = [self.system_prompt] if self.system_prompt else []
        for task in self.tasks.values():
            if task.get('image_path'):
                try:
                    os.remove(task['image_path'])
                except FileNotFoundError:
                    pass
        self.tasks = {}
        if os.path.exists(self.tmp_dir):
            for file in os.listdir(self.tmp_dir):
                os.remove(os.path.join(self.tmp_dir, file))
