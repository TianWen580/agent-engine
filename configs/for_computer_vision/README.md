## Overview
This configuration file is designed to correct species classification results in COCO annotations using a biologist agent powered by the `Qwen/Qwen2.5-VL-3B-Instruct` model. The workflow processes images and their corresponding COCO annotations, ensuring that the classifications align with the allowed species classes.

---

## Genral Parameters

1. **`model_name`**  
   - Refer to the model to initialize agents.
   - Example: You can change to your loacal model path, like `/xxx/Qwen/Qwen2.5-VL-3B-Instruct`. Or you can use online api with `model_name` format in `api_url@api_key@model_name` (there are three parts seperated by `@`. for  example `https://openrouter.ai/api/v1/chat/completions@sk-or-v1-your-key@qwen/qwen2.5-vl-72b-instruct:free`)

2. **`tmp_dir`**  
   - Directory for agent temporary files generated during processing.  
   - Example: `asset/tmp`  

3. **`max_new_tokens`**  
   - Maximum number of tokens the model can generate in its response.  
   - Example: `1024`  
   - Adjust based on the complexity of the task and available computational resources.

---

## Specific Parameters
These parameters are unique to this workflow and must be customized based on your requirements:

1. **`allowed_classes`**  
   - List of species classes that are considered valid during classification.  
   - Example: `["东北虎"]`  

2. **`coco_paths`**  
   - Paths to the COCO annotation files (JSON format) or COCO root folder.  
   - Example: `["asset/coco/东北虎.json"]` or `"asset/coco"`

3. **`images_path`**  
   - Directory containing the images referenced in the COCO annotations.  
   - Example: `asset/coco/images`  

4. **`save_paths`**  
   - Paths where the corrected COCO annotations will be saved, it can be file list or root folder.  
   - Example: `["output/coco/东北虎_refined.json"]` or `output/coco`
