from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

class Event:
    def __init__(self, name, event_type, dependencies=None, logic=None, failure_rate=None):
        self.name = name
        self.event_type = event_type
        self.dependencies = dependencies if dependencies is not None else []
        self.logic = logic
        self.failure_rate = failure_rate
        self.evidence_count = 0  # 初始化证据次数

    def increase_evidence_count(self):
        self.evidence_count += 1  # 增加证据次数

    def increase_failure_rate(self, increase_rate):
        self.failure_rate += increase_rate  # 增加失效率
        self.failure_rate = min(self.failure_rate, 1)  # 确保失效率不超过1

# input_excel_file = 'input_data.xlsx'  # 替换为您的输入文件名
# data = pd.read_excel(input_excel_file)

# 模拟的原始数据
data = {
    'Top_Event': ['TE1', 'TE1', 'TE1', 'TE2', 'TE3', 'TE3'],
    'Intermediate_Event': ['IE1', 'IE1', None, 'IE2', 'IE3', 'IE3'],
    'Basic_Event': ['BE1', 'BE2', 'BE3', 'BE4', 'BE5', 'BE2'],
    'Basic_Event_Rate': [0.03, 0.03, 0.03, 0.06, 0.06, 0.03],
    'Logic': ['OR', 'OR', 'OR', 'AND', 'OR', 'OR']
}

# 构建知识图谱
G = nx.DiGraph()

# 添加节点和边
for i, top_event in enumerate(data['Top_Event']):
    intermediate_event = data['Intermediate_Event'][i]
    basic_event = data['Basic_Event'][i]
    logic = data['Logic'][i]
    rate = data['Basic_Event_Rate'][i]

    # 添加事件节点
    G.add_node(top_event, type='Top_Event')
    if intermediate_event:
        G.add_node(intermediate_event, type='Intermediate_Event')
        # 连接顶事件和中间事件
        G.add_edge(intermediate_event, top_event, logic=logic)

    G.add_node(basic_event, type='Basic_Event', rate=rate)
    if intermediate_event:
        # 连接中间事件和底事件
        G.add_edge(basic_event, intermediate_event, logic=logic)
    else:
        # 如果没有中间事件，则直接连接底事件和顶事件
        G.add_edge(basic_event, top_event, logic=logic)

# 可视化知识图谱
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 8))

# 绘制节点
node_colors = ['lightblue' if G.nodes[n]['type'] == 'Top_Event' else 'lightgreen' if G.nodes[n][
                                                                                         'type'] == 'Intermediate_Event' else 'lightcoral'
               for n in G.nodes]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.9)

# 绘制边
nx.draw_networkx_edges(G, pos)

# 标签
nx.draw_networkx_labels(G, pos)
edge_labels = nx.get_edge_attributes(G, 'logic')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title('知识图谱')
plt.axis('off')
plt.show()

# 转换知识图谱为事件列表
def graph_to_events(graph):
    events = []
    for node in graph.nodes(data=True):
        node_name, node_data = node
        event_type = node_data['type']
        failure_rate = node_data.get('rate', None)

        # 获取该节点的直接前驱（依赖）
        dependencies = list(graph.predecessors(node_name))
        # 尝试获取逻辑关系，如果有多个依赖，则需要确定它们之间的逻辑关系
        # 注意：这里简化处理，假设所有依赖的逻辑关系相同
        if dependencies:
            logic = graph.edges[dependencies[0], node_name]['logic']
        else:
            logic = None

        # 根据事件类型创建 Event 对象
        if event_type in ['Top_Event', 'Intermediate_Event', 'Basic_Event']:
            events.append(
                Event(node_name, event_type, dependencies=dependencies, logic=logic, failure_rate=failure_rate))

    return events


# 转换图谱并构建贝叶斯网络
events = graph_to_events(G)

def build_model(events):
    model = BayesianNetwork()

    for event in events:
        model.add_node(event.name)
        for dependency in event.dependencies:
            model.add_edge(dependency, event.name)

    for event in events:
        cpd = build_cpd(event,events)
        model.add_cpds(cpd)

    if not model.check_model():
        raise ValueError("模型构建失败")

    return model

def calculate_failure_rate(event, events):
    if event.failure_rate is not None:
        return event.failure_rate  # 对于底事件，直接返回其失效率

    if not event.dependencies:
        return None  # 无依赖的事件无法计算失效率

    dependencies = event.dependencies
    failure_rates = []

    for dep_name in dependencies:
        dep_event = next((e for e in events if e.name == dep_name), None)
        if dep_event is None:
            raise ValueError("Dependency event not found: {}".format(dep_name))

        dep_failure_rate = calculate_failure_rate(dep_event, events)
        failure_rates.append(dep_failure_rate)

    if event.logic == "OR":
        return 1 - np.prod([1 - rate for rate in failure_rates])
    elif event.logic == "AND":
        return np.prod(failure_rates)

def build_cpd(event, events):
    failure_rate = calculate_failure_rate(event, events)

    if failure_rate is not None:
        event.failure_rate = failure_rate  # 更新事件的失效率

    if not event.dependencies:
        return TabularCPD(variable=event.name, variable_card=2,
                          values=[[1 - failure_rate], [failure_rate]])

    dependencies = event.dependencies
    evidence_card = [2] * len(dependencies)
    cpd_table = np.zeros((2, np.prod(evidence_card)))

    cpd_table[1, 0] = failure_rate
    cpd_table[0, 0] = 1 - failure_rate

    for i in range(1, np.prod(evidence_card)):
        prob_increase = 0.3 * bin(i).count("1")
        cpd_table[1, i] = min(1, failure_rate + prob_increase)
        cpd_table[0, i] = 1 - cpd_table[1, i]

    return TabularCPD(variable=event.name, variable_card=2,
                      values=cpd_table, evidence=dependencies,
                      evidence_card=evidence_card)


def adaptive_increase_rate(event, events):
    base_increase = min(max(0.01, event.failure_rate), 0.05)

    evidence_factor = 1 + (event.evidence_count / (1 + sum(e.evidence_count for e in events) / len(events)))

    # 查找依赖事件对象
    dependencies = [e for e in events if e.name in event.dependencies]
    if dependencies:
        avg_dep_failure_rate = np.mean([e.failure_rate for e in dependencies])
        avg_dep_evidence_count = np.mean([e.evidence_count for e in dependencies])
        dependency_factor = (avg_dep_failure_rate + avg_dep_evidence_count) / 2
    else:
        dependency_factor = 1

    increase_rate = base_increase * evidence_factor * dependency_factor
    return min(increase_rate, 0.05)


def auto_increase_and_rebuild(evidence_events, events, model):
    # 增加证据事件的失效率
    for evidence_event in evidence_events:
        for event in events:
            if event.name == evidence_event and event.failure_rate is not None:
                event.increase_evidence_count()  # 作为证据时增加次数
                increase_rate = adaptive_increase_rate(event, events)  # 计算动态增加率
                event.increase_failure_rate(increase_rate)

    # 重新计算整个CPD表
    model.remove_cpds(*model.get_cpds())
    for event in events:
        cpd = build_cpd(event, events)  # 假设build_cpd是已经定义好的
        model.add_cpds(cpd)
    assert model.check_model(), "模型重建失败"

def dynamic_inference(model, events, evidence_events, threshold_probability):
    inference = VariableElimination(model)
    events_probabilities = {}

    # 将单一和复合证据统一处理为列表形式
    if not isinstance(evidence_events, list):
        evidence_events = [evidence_events]
    evidence_dict = {}

    # 遍历证据事件，处理复合证据
    for evidence in evidence_events:
        if isinstance(evidence, tuple) or isinstance(evidence, list):
            # 复合证据：将所有证据事件设置为发生
            for ev in evidence:
                evidence_dict[ev] = 1
        else:
            # 单一证据
            evidence_dict[evidence] = 1

    # 对于复合证据，决定可推理的事件类型
    inferable_types = set()
    for evidence in evidence_dict.keys():
        event_obj = next((e for e in events if e.name == evidence), None)
        if event_obj:
            if event_obj.event_type == "Top_Event":
                inferable_types.update(["Intermediate_Event", "Basic_Event"])
            elif event_obj.event_type == "Intermediate_Event":
                inferable_types.add("Basic_Event")
            else:  # Basic_Event
                inferable_types.update(["Intermediate_Event", "Top_Event"])

    # 打印推理前的证据事件失效率并推理
    for evidence in evidence_dict.keys():
        event_obj = next((event for event in events if event.name == evidence), None)
        if event_obj:
            print(f"推理前 '{evidence}' 的失效率: {event_obj.failure_rate}")

    # 推理非证据事件的发生概率
    for event in events:
        if event.name not in evidence_dict and event.event_type in inferable_types:
            result = inference.query(variables=[event.name], evidence=evidence_dict)
            probability = result.values[1]
            if probability > threshold_probability:
                events_probabilities[event.name] = probability

    # 对每个作为证据的事件增加失效率并重建模型
    auto_increase_and_rebuild(list(evidence_dict.keys()), events, model)

    # 打印推理后的证据事件失效率
    for evidence in evidence_dict.keys():
        event_obj = next((event for event in events if event.name == evidence), None)
        if event_obj:
            print(f"推理后 '{evidence}' 的失效率: {event_obj.failure_rate}")

    return events_probabilities

model = build_model(events)

# 检查模型是否正确构建
assert model.check_model(), "模型构建失败"

# 显示构建好的CPD表
def display_cpds(model):
    for cpd in model.get_cpds():
        print("CPD of", cpd.variable)
        print(cpd)  # 打印CPD表
        print("\n")

display_cpds(model)


import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一种常用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from plotly.subplots import make_subplots
import time

def plot_probabilities_subplot(fig, probabilities, title, row, col, rows, cols):
    labels = list(probabilities.keys())
    values = [probabilities[key] for key in labels]

    # 突出显示概率最大的事件
    max_prob = max(values)
    pull = [0.1 if value == max_prob else 0 for value in values]

    # 设置饼图的悬浮信息模板
    hoverinfo = 'label+percent+value'
    hovertemplate = '%{label}<br>概率: %{value:.5f}<br>占比: %{percent}'

    pie = go.Pie(labels=labels, values=values, pull=pull, hoverinfo=hoverinfo, hovertemplate=hovertemplate,
                 marker=dict(line=dict(color='#000000', width=2)),  # 添加边框
                 showlegend=True,  # 显示图例
                 textinfo='none')  # 饼图块上不显示文本

    fig.add_trace(pie, row=row, col=col)

    # 添加图表的标题
    fig.update_layout(title_text="随时间变化的事件概率分析", title_x=0.5)

    # 设置图例的位置和样式
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.1,
        xanchor="center",
        x=0.5
    ))

    # 计算注释的正确位置
    x_pos = (col - 0.7) / cols  # 将x坐标定位在列的中间
    if row == 1:
        y_pos = (rows - row) / rows  # 第一行的注释放在饼图上方
    else:
        y_pos = (rows - row - 0.1) / rows  # 第二行的注释放在饼图下方

    fig.add_annotation(
        xref="paper", yref="paper",
        x=x_pos, y=y_pos,
        text=title,
        showarrow=False,
        font=dict(size=10, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderpad=4,
        align="center"
    )

def create_line_chart(failure_rates_over_time):
    fig = go.Figure()

    # 定义一组美观的颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (event_name, rates) in enumerate(failure_rates_over_time.items()):
        color = colors[i % len(colors)]  # 循环使用颜色
        fig.add_trace(go.Scatter(
            x=list(range(len(rates))),
            y=rates,
            mode='lines+markers',
            name=event_name,
            line=dict(color=color, width=2),  # 自定义线条颜色和宽度
            marker=dict(color=color, size=8, line=dict(color='white', width=1)),  # 自定义标记样式
            hoverinfo='text',
            text=[f"{event_name}: {rate:.2f}" for rate in rates]  # 设置悬停文本
        ))

    fig.update_layout(
        title="<b>各底事件失效率随时间的变化</b>",  # 使用HTML标签增加标题的样式
        xaxis=dict(title="推理步骤", showgrid=False),  # 隐藏网格线
        yaxis=dict(title="失效率", range=[0, 1]),  # 设置y轴范围为0到1
        legend=dict(title="底事件", x=1.05, y=1, bordercolor="Black", borderwidth=1),  # 调整图例位置和样式
        margin=dict(l=20, r=20, t=50, b=20),  # 调整边距以适应显示
        template="plotly_white",  # 使用内置模板
    )

    fig.show()

def create_stacked_area_chart(failure_rates_over_time):
    fig = go.Figure()

    for event_name, rates in failure_rates_over_time.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(rates))),
            y=rates,
            mode='lines',
            fill='tonexty',  # 填充到下一个轨迹
            name=event_name
        ))

    fig.update_layout(
        title="各底事件失效率随时间的累积变化",
        xaxis_title="推理步骤",
        yaxis_title="失效率",
        legend_title="底事件",
        margin=dict(l=0, r=0, t=30, b=0)
    )

    fig.show()


import plotly.graph_objects as go


def create_animated_bar_chart(failure_rates_over_time):
    # 初始化图形对象
    fig = go.Figure()

    # 设置动画的初始状态
    fig.add_trace(go.Bar(
        x=list(failure_rates_over_time.keys()),
        y=[0] * len(failure_rates_over_time),
        text=[0] * len(failure_rates_over_time),
        textposition='auto',
    ))

    # 定义动画帧
    frames = [
        go.Frame(
            data=[
                go.Bar(
                    x=list(failure_rates_over_time.keys()),
                    y=rates,
                    text=rates,  # 在柱状图上显示具体的失效率数值
                    textposition='auto',
                )
            ],
            name=str(step)
        )
        for step, rates in enumerate(zip(*failure_rates_over_time.values()))
    ]

    # 更新布局，添加标题和动画控件
    fig.update_layout(
        title="各底事件失效率随时间的变化",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="播放",
                          method="animate",
                          args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}])],
            direction="left",
            pad={"r": 10, "t": 87},
            showactive=True,
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top"
        )],
        sliders=[{
            "steps": [{"args": [[f.name],
                                {"frame": {"duration": 500, "redraw": True}, "mode": "immediate", "fromcurrent": True}],
                       "label": str(k), "method": "animate"} for k, f in enumerate(frames)],
        }]
    )

    # 添加帧
    fig.frames = frames

    # 美化
    fig.update_traces(marker_color='RoyalBlue', marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)
    fig.update_layout(template="plotly_white")

    fig.show()


def dynamic_visualization_subplot(model, events, evidence_events):
    rows, cols = 2, 3
    plots_per_page = rows * cols
    page_counter = 1  # 用于跟踪当前是第几页

    # 初始化存储每个底事件失效率变化的字典
    basic_event_failure_rates_over_time = {event.name: [] for event in events if event.event_type == "Basic_Event"}

    # 确定总共需要多少页
    total_pages = len(evidence_events) // plots_per_page + (1 if len(evidence_events) % plots_per_page > 0 else 0)

    # 遍历每一页
    for page in range(total_pages):
        # 对于每一页，创建一个新的图表
        fig = make_subplots(rows=rows, cols=cols,
                            specs=[[{'type': 'domain'} for _ in range(cols)] for _ in range(rows)],
                            subplot_titles=[f"推理结果 - {evidence_events[page * plots_per_page + i]}" for i in
                                            range(min(plots_per_page, len(evidence_events) - page * plots_per_page))])

        # 在当前页上为每个证据事件添加子图
        for i in range(plots_per_page):
            evidence_index = page * plots_per_page + i
            if evidence_index >= len(evidence_events):
                break  # 如果所有证据事件都已处理，跳出循环

            evidence_event = evidence_events[evidence_index]
            row = (i // cols) + 1
            col = (i % cols) + 1

            # 进行推理并绘制饼图
            # 在dynamic_visualization_subplot中，调用dynamic_inference时传入混合证据列表
            probabilities = dynamic_inference(model, events, [evidence_event], 0)

            print(f"推理结果 - {evidence_event}: {probabilities}")  # 打印检查概率值
            plot_probabilities_subplot(fig, probabilities, f"证据事件 '{evidence_event}'的推理结果", row, col, rows, cols)

            # 收集每个底事件的当前失效率
            for event in events:
                if event.event_type == "Basic_Event":
                    basic_event_failure_rates_over_time[event.name].append(event.failure_rate)

        # 更新布局并显示当前页
        fig.update_layout(height=700, showlegend=True,
                          title_text=f"随证据变化的事件概率分析 - 页 {page + 1} / {total_pages}")
        fig.show()
        time.sleep(2)  # 用于演示，实际使用时可移除
        file_name = f"dynamic_visualization_page_{page_counter}.html"
        fig.write_html(file_name)
        print(f"Page {page_counter} saved as {file_name}")
        page_counter += 1

    # 在完成所有循环后，用收集到的数据生成底事件失效率的动态图表
    create_line_chart(basic_event_failure_rates_over_time)
    create_animated_bar_chart(basic_event_failure_rates_over_time)


def generate_mixed_evidence(events, k):
    # 示例: 随机选择事件生成证据列表，包括单个事件和事件对
    all_events = [e.name for e in events]
    mixed_evidence = []

    # 随机添加单个事件
    for _ in range(k // 2):
        mixed_evidence.append(random.choice(all_events))

    # 随机添加事件对
    for _ in range(k // 2):
        ev1, ev2 = random.sample(all_events, 2)
        mixed_evidence.append((ev1, ev2))

    return mixed_evidence[:k]


# 生成混合证据列表
mixed_evidence_events = generate_mixed_evidence(events, 10)  # 数量根据需求调整
print("Mixed Evidence Events:", mixed_evidence_events)

# 使用混合证据进行可视化
dynamic_visualization_subplot(model, events, mixed_evidence_events)
