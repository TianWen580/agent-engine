# __init__.py for workflow module
"""
AgentEngine workflow module.

This module provides workflow management capabilities for the AgentEngine,
allowing for the definition, execution, and monitoring of agent workflows.
"""
from .base import BaseWorkflow
from .for_crawler.baike_crawl_workflow import BaikeSpeciesWorkflow
from .for_crawler.baike_species_name_translate_workflow import BaikeSpeciesNameTranslateWorkflow
from .for_database.database_query_workflow import DatabaseQueryWorkflow
from .for_computer_vision.coco_cls_checker_workflow import COCOClassCheckerWorkflow

__all__ = [
    'BaseWorkflow',
    'BaikeSpeciesWorkflow',
    'BaikeSpeciesNameTranslateWorkflow',
    'DatabaseQueryWorkflow',
    'COCOClassCheckerWorkflow'
]
