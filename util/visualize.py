import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


# 读取日志文件
def parse_log_file(file_path):
    epochs = []
    train_losses = []

    with open(file_path, "r") as f:
        for line in f:
            # 使用正则表达式匹配数字
            match = re.search(r"Epoch (\d+), Train Loss: ([\d.]+)", line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                epochs.append(epoch)
                train_losses.append(train_loss)

    return epochs, train_losses


# 绘制曲线
def plot_loss_curves(log_file, save_path):
    epochs, train_losses = parse_log_file(log_file)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=0.5)

    plt.title("Training Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=5)
    # plt.grid(True, linestyle="--", alpha=0.7)

    plt.xticks(np.arange(0, max(epochs), 1000))
    plt.xlim(0, max(epochs))
    # plt.ylim(0, max(train_losses))
    plt.ylim(0, 10)

    # 标记最低损失点
    # min_train_loss = min(train_losses)
    # min_idx = train_losses.index(min_train_loss)
    # plt.scatter(epochs[min_idx], min_train_loss, s=10, c="red", zorder=5)
    # plt.annotate(
    #     f"Best Train Loss: {min_train_loss:.4f}\n epoch:{min_idx}",
    #     xy=(epochs[min_idx], min_train_loss),
    #     xytext=(epochs[min_idx], min_train_loss + 0.01),
    #     ha="left",
    #     fontsize=10,
    #     arrowprops=dict(facecolor="black", shrink=0.05),
    # )

    # plt.tight_layout()
    # plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def draw_confusion_matrix(labels, preds, save_path, avg_val_loss):
    task_names = [
        "flySlag",
        "bubbleSlag",
        "stratification",
        "roughness",
        "breach",
        "fluidSlag",
        "slag",
        "scum",
        "burrs",
        "overBreach",
    ]

    plt.figure(figsize=(10, 15))

    for i in range(10):
        # 混淆矩阵 ---
        plt.subplot(5, 2, i + 1)
        cm = confusion_matrix(labels[i], preds[i], labels=np.arange(6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=np.arange(6),
            yticklabels=np.arange(6),
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 8},
        )
        plt.title(f"Task {i+1}: {task_names[i]}", fontsize=10)
        plt.xlabel("Predicted", fontsize=8)
        plt.ylabel("Actual", fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

    # 调整布局并保存
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    log_file = f"/home/chenjiayi/code/params-tuning/log/type0/train_loss.log"
    save_path = f"/home/chenjiayi/code/params-tuning/log/type0"
    plot_loss_curves(log_file, save_path)
