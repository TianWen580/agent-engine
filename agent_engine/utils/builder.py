import yaml
from typing import Any, Type
import importlib

def create_workflow(config: str) -> Any:
    """
    Create a workflow object from a YAML specification.
    
    Args:
        yaml_content: The YAML content specifying the workflow
        
    Returns:
        A fully cfgured workflow object
    """
    # Parse the YAML content
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Get the workflow type
    workflow_type_str = cfg['workflow']['type']
    if not workflow_type_str:
        raise ValueError("Workflow type must be specified in YAML")
    
    # Dynamically import the workflow class
    workflow_class = import_class(workflow_type_str)
    
    # Initialize workflow builder
    builder = WorkflowBuilder()
    builder.set_class(workflow_class)
    
    return builder.build(workflow_class, config)

def import_class(class_path: str) -> Type:
    """
    Dynamically import a class from its string path.
    
    Args:
        class_path: Full import path of the class (e.g., 'engine.workflow.baike_crawl_workflow.BaikeSpeciesWorkflow')
        
    Returns:
        The class object
    """
    try:
        if '.' in class_path:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        else:
            # If no module path is provided, try to find the class in common locations
            for module_path in ["engine.workflow", "engine.agent"]:
                try:
                    module = importlib.import_module(module_path)
                    if hasattr(module, class_path):
                        return getattr(module, class_path)
                except (ImportError, AttributeError):
                    continue
            
            raise ImportError(f"Could not find class {class_path}")
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {class_path}: {e}")

class WorkflowBuilder:
    """
    Builder class for creating and cfguring workflow instances.
    """
    def __init__(self):
        self._class = None
        self._instance = None
        self._name = None
        self._cfg = {}
    
    def set_class(self, workflow_class: Type) -> 'WorkflowBuilder':
        """Set the workflow class and initialize an instance"""
        self._class = workflow_class
        return self
        
    def build(self, workflow_class: Type, cfg) -> Any:
        self._instance = workflow_class(cfg)
        return self._instance
