import os
import uuid
import requests
import base64
from typing import Optional
from PIL import Image
from agent_engine.agent import BaseChatEngine


class ContextualChatEngine(BaseChatEngine):
    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        tmp_dir: str = "asset/tmp",
        max_new_tokens: int = 512,
        vllm_cfg: Optional[dict] = None
    ):
        super().__init__(
            model_name,
            system_prompt,
            tmp_dir,
            max_new_tokens,
            vllm_cfg
        )

    def generate_response(self, prompt: str, img_path: Optional[str] = None) -> dict:
        job_id = str(uuid.uuid4())

        if self.is_online:
            content = []

            if self.model_config.supports_images and img_path:
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

            content.append({
                "type": "text",
                "text": prompt
            })

            messages = self.context.copy()
            messages.append({
                "role": "user",
                "content": content
            })

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
            if self.model_config.supports_images and img_path:
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
        """
        Generic task processing method compatible with AutoModelForCausalLM.
        Handles text-only and multi-modal models dynamically.
        """
        try:
            task = self.tasks[job_id]
            img_path = task['image_path']
            prompt = task['prompt']
            messages = self.context.copy()

            # Prepare user content based on model capabilities
            user_content = None
            if img_path and self.model_config.supports_images:
                user_content = []
                img = Image.open(img_path).convert("RGB")
                user_content.append({
                    "type": "image",
                    "image": img,
                })
                user_content.append({
                    "type": "text",
                    "text": prompt
                })
            else:
                user_content = prompt

            # Add user message to the conversation
            user_msg = {"role": "user", "content": user_content}
            messages.append(user_msg)

            # Process inputs using the processor (if available)
            if self.processor:
                try:
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if self.model_config.supports_images and img_path:
                        inputs = self.processor(
                            text=[text],
                            images=[item["image"] for item in user_content if item["type"] == "image"],
                            return_tensors="pt",
                            padding=True,
                        ).to(self.model.device)
                    else:
                        inputs = self.processor(
                            text=[text],
                            return_tensors="pt",
                            padding=True,
                        ).to(self.model.device)
                except Exception as e:
                    raise ValueError(f"Failed to process inputs: {e}")
            else:
                # If no processor is available, use raw text input
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)

            # Generate response
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

            # Decode the generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Update task status and context
            self.tasks[job_id]['status'] = 'completed'
            self.tasks[job_id]['result'] = output_text
            if self.model_config.supports_images:
                ai_msg = [{"type": "text", "text": output_text}]
            else:
                ai_msg = output_text
            self.context.append({"role": "assistant", "content": ai_msg})
            return self.tasks[job_id]

        except Exception as e:
            self.tasks[job_id]['status'] = 'error'
            self.tasks[job_id]['result'] = str(e)
            print(f"[AGENT] Error processing task {job_id}: {e}")
            if img_path and os.path.exists(img_path):
                os.remove(img_path)
            return self.tasks[job_id]