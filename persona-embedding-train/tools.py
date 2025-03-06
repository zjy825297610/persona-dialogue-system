import json
import random
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.styles import Font


def read_from_json(input_file, is_shuffle):
    data_list = []
    with open(input_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            data_list.append(json.loads(line))
    if is_shuffle:
        random.shuffle(data_list)
        return data_list
    return data_list

def save_as_json(input_list, save_path):
    with open(save_path, 'w', encoding='utf-8') as writer:
        for line in input_list:
            writer.write(json.dumps(line, ensure_ascii=False)+'\n')

def save_results_excel(data, excel_filename):
    rows = []
    for item in data:
        for key, result in item.items():
            row = {"Model+Data": key}
            row.update(result)
            rows.append(row)

    df = pd.DataFrame(rows)

    # 保存为 Excel 并加粗最高值
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
        workbook = writer.book
        worksheet = writer.sheets["Results"]

        # 加粗每列的最大值
        for col_idx, col_name in enumerate(df.columns[1:], start=2):  # 从第2列开始（跳过 Model+Data）
            max_row = df[col_name].idxmax() + 2  # 找到最大值的行索引（加2是因为 Excel 索引从第2行开始）
            cell = worksheet.cell(row=max_row, column=col_idx)
            cell.font = Font(bold=True)

    print(f"表格已保存为 Excel 文件：{excel_filename}")

def save_results_pic(data, output_path):
    # 转换为 DataFrame
    rows = []
    for item in data:
        for key, result in item.items():
            row = {"Model+Data": key}
            row.update(result)
            rows.append(row)

    df = pd.DataFrame(rows)

    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 4))  # 设置图片尺寸
    ax.axis('tight')
    ax.axis('off')

    # 创建表格数据
    table_data = df.values.tolist()
    columns = df.columns.tolist()

    # 将每列的最大值加粗处理
    for col_idx in range(1, len(columns)):  # 从第1列开始（跳过 Model+Data 列）
        max_row_idx = df[columns[col_idx]].idxmax()
        table_data[max_row_idx][col_idx] = f"**{table_data[max_row_idx][col_idx]:.4f}**"

    # 绘制表格
    table = plt.table(cellText=[[f"{v:.4f}" if isinstance(v, float) else v for v in row] for row in df.values],
                      colLabels=columns, cellLoc='center', loc='center')

    # 自动调整列宽和字体大小
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(columns))))

    # 修改单元格字体加粗样式
    for col_idx in range(1, len(columns)):  # 跳过第一列
        max_row_idx = df[columns[col_idx]].idxmax()
        cell = table[max_row_idx + 1, col_idx]  # +1 是因为标题行也被计入
        cell.set_text_props(fontweight='bold')  # 设置字体加粗

    # 保存图片
    plt.savefig(f"{output_path}", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"表格已保存为图片：{output_path}")