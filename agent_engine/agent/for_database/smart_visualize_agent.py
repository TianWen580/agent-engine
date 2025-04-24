import os
import json
import re
from typing import List, Dict, Optional
from pyecharts.charts import Bar, Line, Pie
from pyecharts import options as opts
from agent_engine.agent import ContextualChatEngine


class SmartVisualizeAgent:
    """由语言模型驱动的智能可视化代理"""

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        language: str = "english",
        tmp_dir: str = "asset/tmp",
        max_new_tokens: int = 1024,
        vllm_cfg: Optional[dict] = None
    ):
        self.chat_engine = ContextualChatEngine(
            model_name=model_name,
            system_prompt=system_prompt,
            language=language,
            tmp_dir=tmp_dir,
            max_new_tokens=max_new_tokens,
            vllm_cfg=vllm_cfg
        )
        self.save_dir = "output/visuals"
        os.makedirs(self.save_dir, exist_ok=True)

    def generate_visuals(
        self,
        user_query: str,
        data_structure: Dict[str, str],
        data_sample: List[Dict]
    ) -> List[str]:
        """生成可视化图表"""
        prompt = f"""
用户查询：{user_query}
数据结构：{json.dumps(data_structure, ensure_ascii=False)}
数据样例：{json.dumps(data_sample[:2], ensure_ascii=False)}

请生成3种最合适的ECharts配置（JSON格式），要求严格遵循以下规范：

一、基础规范
1. 必须包含的顶级字段：
- type: 图表类型（bar/line/pie）
- title: 包含text/subtext/pos_left等子字段
- xAxis/yAxis: 必须声明type（'category'/'value'）
- series: 包含name/type/data等字段
- tooltip: 声明trigger类型
- legend: 声明show/data等属性

2. 命名规范：
- 所有字段使用蛇形命名法（如：axis_label）
- 禁止使用驼峰命名（如：axisLabel将导致错误）

二、数据绑定规范
1. xAxis数据必须与数据结构中的分类字段对应
2. series.data必须为数值数组，与数据样例中的度量字段对应
3. pie类型必须使用data中的name/value结构

三、可视化增强规范
1. 必须包含以下交互配置：
- tooltip.formatter: 自定义显示格式
- visual_map: 当需要数据映射时添加
- toolbox.feature: 保存/缩放等工具

2. 美观性要求：
- 使用渐变色配置（linear-gradient）
- 坐标轴刻度对齐方式
- 图例位置自动适配
- 响应式布局配置

四、输出规范
1. 每个配置必须通过JSON Schema验证：
{{
"type": "object",
"properties": {{
    "type": {{"type": "string"}},
    "title": {{
    "type": "object",
    "properties": {{
        "text": {{"type": "string"}},
        "subtext": {{"type": "string"}},
        "pos_left": {{"type": "string"}}
    }},
    "required": ["text"]
    }},
    "xAxis": {{
    "type": "object",
    "properties": {{
        "type": {{"enum": ["category", "value"]}},
        "data": {{"type": "array"}}
    }},
    "required": ["type"]
    }},
    "yAxis": {{"type": "object"}},
    "series": {{
    "type": "array",
    "items": {{
        "type": "object",
        "properties": {{
        "name": {{"type": "string"}},
        "type": {{"type": "string"}},
        "data": {{"type": "array"}}
        }},
        "required": ["name", "type", "data"]
    }}
    }}
}},
"required": ["type", "title", "xAxis", "yAxis", "series"]
}}

2. 输出3个```json包裹的代码块，每个配置包含：
- 业务相关的标题和轴标签
- 与数据样例匹配的数据结构
- 完整的交互功能配置
- 符合企业级审美的样式
        """

        response = self.chat_engine.generate_response(prompt)
        code_blocks = re.findall(
            r"```json(.*?)```", response['result'], re.DOTALL)

        file_paths = []
        for idx, config in enumerate(code_blocks[:3]):
            try:
                chart_config = json.loads(config.strip())
                file_name = f"visual_{user_query[:10]}_{idx}.html"
                file_path = os.path.join(self.save_dir, file_name)

                # 实际生成ECharts图表
                chart = self._render_echarts(chart_config)
                chart.render(file_path)
                file_paths.append(file_path)
            except Exception as e:
                self.chat_engine.console.print(f"[ERROR] 图表生成失败: {str(e)}")

        return file_paths

    def _render_echarts(self, config):
        """
        根据完整的 ECharts 配置渲染图表
        :param config: 完整的 ECharts 配置字典
        :return: 渲染后的 pyecharts 图表对象
        """
        try:
            # 验证配置完整性
            self._validate_config(config)

            # 提取图表类型，默认为 'bar'
            chart_type = config.get('type', 'bar').lower()

            # 初始化图表对象
            if chart_type == 'bar':
                chart = Bar()
            elif chart_type == 'line':
                chart = Line()
            elif chart_type == 'pie':
                chart = Pie()
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            # 设置全局配置项
            global_opts = {}
            if 'title' in config:
                title_config = config['title']
                global_opts['title_opts'] = opts.TitleOpts(
                    title=title_config.get('text'),
                    subtitle=title_config.get('subtitle'),
                    pos_left=title_config.get('posLeft'),
                    pos_right=title_config.get('posRight'),
                    pos_top=title_config.get('posTop'),
                    pos_bottom=title_config.get('posBottom')
                )

            if 'tooltip' in config:
                tooltip_config = self._convert_keys_to_snake_case(
                    config['tooltip'])
                if 'axis_pointer' in tooltip_config:
                    # 特殊处理 axisPointer
                    axis_pointer_config = tooltip_config.pop('axis_pointer')
                    tooltip_config['axispointer_opts'] = opts.AxisPointerOpts(
                        **axis_pointer_config)
                global_opts['tooltip_opts'] = opts.TooltipOpts(
                    **tooltip_config)

            if 'legend' in config:
                legend_config = self._convert_keys_to_snake_case(
                    config['legend'])
                global_opts['legend_opts'] = opts.LegendOpts(**legend_config)

            if 'xAxis' in config:
                x_axis_config = self._convert_keys_to_snake_case(
                    config['xAxis'])
                global_opts['xaxis_opts'] = opts.AxisOpts(**x_axis_config)

            if 'yAxis' in config:
                y_axis_config = self._convert_keys_to_snake_case(
                    config['yAxis'])
                global_opts['yaxis_opts'] = opts.AxisOpts(**y_axis_config)

            if 'grid' in config:
                grid_config = self._convert_keys_to_snake_case(config['grid'])
                global_opts['grid_opts'] = opts.GridOpts(**grid_config)

            chart.set_global_opts(**global_opts)

            # 添加数据系列
            series = config.get('series', [])
            for serie in series:
                if chart_type in ['bar', 'line']:
                    # 确保xAxis数据是字符串列表
                    x_data = [str(x) for x in serie.get('xAxisData', [])]
                    chart.add_xaxis(x_data)

                    # 处理yAxis数据
                    y_data = serie.get('data', [])
                    if not all(isinstance(d, (int, float)) for d in y_data):
                        raise ValueError("Y轴数据必须为数值类型")

                    chart.add_yaxis(
                        serie.get('name', ''),
                        y_data,
                        **serie.get('options', {})
                    )
                elif chart_type == 'pie':
                    chart.add(
                        serie.get('name', ''),
                        [(item['name'], item['value'])
                         for item in serie.get('data', [])],
                        **serie.get('options', {})
                    )

            # 返回渲染完成的图表对象
            return chart

        except Exception as e:
            error_msg = f"渲染失败: {str(e)}\n配置内容: {json.dumps(config, indent=2)}"
            self.chat_engine.console.print(f"[ERROR] {error_msg}")
            raise RuntimeError(error_msg)

    def _validate_config(self, config):
        """验证配置是否符合要求"""
        required_fields = ['type', 'series']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"缺失必要字段: {field}")

        if config['type'] not in ['bar', 'line', 'pie']:
            raise ValueError(f"不支持的图表类型: {config['type']}")

    def _convert_keys_to_snake_case(self, config: dict) -> dict:
        """将驼峰命名转换为pyecharts使用的蛇形命名，并递归处理嵌套字典"""
        new_config = {}
        for key, value in config.items():
            new_key = re.sub(r'(?<!^)(?=[A-Z])', '_', key).lower()
            if isinstance(value, dict):
                # 递归处理嵌套字典
                new_config[new_key] = self._convert_keys_to_snake_case(value)
            else:
                new_config[new_key] = value
        return new_config
