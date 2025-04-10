```shell
╭──────  AGENT ENGINE  ──────╮╮
│  ░▒▓░▒▓░▒▓░▒▓░▒▓░▒▓░▒▓░▒   ││  Help you creating things better ~
╰────────────────────────────╯╯
```
# AgentEngine
AgentEngine is a powerful and flexible framework designed to build, manage, and deploy intelligent agents for various applications.

## Features

- Modular architecture for easy customization.
- Scalable and efficient performance.
- Comprehensive API for seamless integration.

## Installation

```bash
git clone https://github.com/your-repo/agent-engine.git
cd agent_engine
pip install -e .
```
- requirements installation
```bash
pip install -r requirements.txt
```

## Run workflow in project
> example in [run.py](run.py) for database query
- create new python script
- import `create_workflow` function from `agent_engine.utils`
```python
from agent_engine.utils import create_workflow
```
- then init workflow and execute it in this way (example config in [configs/for_computer_vision/coco_cls_checker.yaml](configs/for_computer_vision/coco_cls_checker.yaml))
```python
workflow = create_workflow(config="configs/for_computer_vision/coco_cls_checker.yaml")
workflow.execute()
```
- [CONFIG] common customization:
    - `model_name: Qwen/Qwen2.5-VL-3B-Instruct`: you can change to your loacal model path, or you can use online api with `model_name` format in `api_url@api_key@model_name` (there are three parts seperated by `@`. for  example `https://openrouter.ai/api/v1/chat/completions@sk-or-v1-your-key@qwen/qwen2.5-vl-72b-instruct:free`)
- every workflow has it's special config parameters, you can check the instruction in every config's directary
- finally, you can check the output in your terminal as like:
![example_png](asset/example_coco_cls_checker.png)
- okey okey, let's deep more into the `agent-engine` !!!

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.