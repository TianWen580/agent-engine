from agent_engine.utils import create_workflow

workflow = create_workflow(config="my_configs/for_computer_vision/coco_cls_checker.yaml")
workflow.execute()