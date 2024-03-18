import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
from datasets.data_generator import DataGenerator

def on_model_change(event):
    # 清除子模型的选项并隐藏下拉列表和确定按钮
    sub_model_combobox['values'] = []
    sub_model_label.grid_remove()
    sub_model_combobox.grid_remove()

    # 如果选中的是Pytorch_model，展示子模型的选择
    if model_combobox.get() == 'Pytorch_model':
        sub_model_combobox['values'] = ['LSTM', 'GRU']
        sub_model_label.grid()
        sub_model_combobox.grid()
        sub_model_combobox.current(0)

def on_confirm():
    selected_model = model_combobox.get()
    selected_sub_model = sub_model_combobox.get() if selected_model == 'Pytorch_model' else ''
    epochs = epochs_entry.get()

    try:
        epochs_int = int(epochs)
        assert epochs_int > 0

        # 准备命令行参数
        cmd = [
            "python", "miniPro_interface.py",
            "--model", selected_model,
            "--algorithm", selected_sub_model,
            "--epochs", str(epochs)
        ]
        if not selected_sub_model:  # 如果没有选择子模型，不要包含空的子模型参数
            cmd = cmd[:-2] + cmd[-1:]

        # 运行minPro.py并捕获输出
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        # 显示minPro.py的标准输出
        output_text.insert(tk.END, stdout)
        output_text.see(tk.END)

        # 显示错误输出
        if stderr:
            output_text.insert(tk.END, "错误:\n" + stderr)
            output_text.see(tk.END)

    except (ValueError, AssertionError):
        output_text.insert(tk.END, "Epochs必须是一个正整数且大于0。\n")




# 创建主窗口
root = tk.Tk()
root.title("TensorFlow Model Training")

# 创建模型选择的下拉列表
model_label = tk.Label(root, text="选择模型:")
model_label.grid(column=0, row=0, sticky=tk.W, padx=10, pady=5)
model_list = ['Pytorch_model', 'keras_lstm_model', 'RandomForest_model', 'SVM_model']
model_combobox = ttk.Combobox(root, values=model_list, state="readonly", width=40)
model_combobox.grid(column=1, row=0, padx=10, pady=5)
model_combobox.current(0)
model_combobox.bind('<<ComboboxSelected>>', on_model_change)

# 子模型选择的下拉列表（默认隐藏）
sub_model_label = tk.Label(root, text="选择子模型:")
sub_model_label.grid(column=0, row=1, sticky=tk.W, padx=10, pady=5)
sub_model_label.grid_remove()
sub_model_combobox = ttk.Combobox(root, state="readonly", width=40)
sub_model_combobox.grid(column=1, row=1, padx=10, pady=5)
sub_model_combobox.grid_remove()

# Epochs输入框
epochs_label = tk.Label(root, text="输入Epochs:")
epochs_label.grid(column=0, row=2, sticky=tk.W, padx=10, pady=5)
epochs_entry = tk.Entry(root, width=43)
epochs_entry.grid(column=1, row=2, padx=10, pady=5)

# 确定按钮
confirm_button = tk.Button(root, text="确定", command=on_confirm)
confirm_button.grid(column=0, row=3, columnspan=2, pady=10)

# 输出文本框
output_text = scrolledtext.ScrolledText(root, width=60, height=10)
output_text.grid(column=0, row=4, columnspan=2, padx=10, pady=10)

root.mainloop()
