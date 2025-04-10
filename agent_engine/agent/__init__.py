"""
Agent module for the AgentEngine project.

This module provides the agent-related functionality and interfaces.
"""

# Import important classes/functions from submodules that should be accessible
# when importing from the agent module
# For example:
# from .agent import Agent
# from .types import AgentType
# from .manager import AgentManager

# Define what should be available when someone does `from agent import *`
from .base import BaseChatEngine
from .context_agent import ContextualChatEngine
from .for_crawler.baike_research_agent import BaikeResearchAgent
from .for_crawler.baike_species_name_translate_agent import BaikeSpeciesNameTranslateAgent
from .for_database.database_query_agent import DatabaseQueryAgent
from .for_database.smart_visualize_agent import SmartVisualizeAgent
from .for_computer_vision.coco_cls_checker_agent import COCOClassCheckerAgent

__all__ = [
    'BaseChatEngine',
    'ContextualChatEngine',
    'BaikeResearchAgent',
    'BaikeSpeciesNameTranslateAgent',
    'DatabaseQueryAgent',
    'SmartVisualizeAgent',
    'COCOClassCheckerAgent',
]

# Version information
__version__ = '0.1.0'