import os   
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
import nltk
from nltk.corpus import stopwords


# # === Load JSON Data ===
# with open("your_file.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# === Create Output Folder ===
output_folder = "report_images"
os.makedirs(output_folder, exist_ok=True)

def get_output_path(unique_id, patient_id, filename):
    subfolder = f"{unique_id}_{patient_id.replace(' ', '_')}"
    folder_path = os.path.join("report_images", subfolder)
    os.makedirs(folder_path, exist_ok=True)
    return os.path.join(folder_path, filename)

# def plot_emotions(emotion_dict):
#     labels = list(emotion_dict.keys())
#     sizes = list(emotion_dict.values())
#     plt.figure(figsize=(6, 6))
#     plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
#     plt.title("Top Emotions")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, "emotions_pie.png"))
#     plt.close()

# def plot_emotions(emotion_dict):
#     labels = list(emotion_dict.keys())
#     sizes = list(emotion_dict.values())

#     # Define color palette (use Set3 for medical themes)
#     cmap = plt.get_cmap("Dark2") # Set3
#     colors = [cmap(i) for i in range(len(labels))]

#     # Explode each slice slightly
#     explode = [0.05] * len(labels)

#     plt.figure(figsize=(8, 6))
#     wedges, texts, autotexts = plt.pie(
#         sizes,
#         explode=explode,
#         colors=colors,
#         autopct="%1.1f%%",
#         startangle=140,
#         pctdistance=0.75,
#         textprops={'fontsize': 11, 'color': 'white', 'weight': 'bold'},
#         wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
#     )

#     # Add center circle (donut style)
#     centre_circle = plt.Circle((0, 0), 0.50, fc='white')
#     plt.gca().add_artist(centre_circle)

#     # Add title and legend outside the chart
#     plt.title("Top Emotions Detected", fontsize=16, fontweight="bold", pad=20)
#     plt.legend(wedges, labels, title="Emotions", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11)

#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, "emotions_pie.png"), bbox_inches="tight")
#     plt.close()
#     print(f"Saved emotion pie chart: {os.path.join(output_folder, 'emotions_pie.png')}")


def plot_emotions(emotion_dict, unique_id, patient_id):
    labels = list(emotion_dict.keys())
    sizes = list(emotion_dict.values())

    cmap = plt.get_cmap("Dark2")
    colors = [cmap(i % 8) for i in range(len(labels))]
    explode = [0.05] * len(labels)

    plt.figure(figsize=(8, 6), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    wedges, texts, autotexts = plt.pie(
        sizes,
        explode=explode,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        textprops={'fontsize': 11, 'color': 'white', 'weight': 'bold'},
        wedgeprops={'linewidth': 1, 'edgecolor': 'black'}
    )

    centre_circle = plt.Circle((0, 0), 0.50, fc='#222222')
    ax.add_artist(centre_circle)

    plt.title("🧠 Top Emotions Detected", fontsize=16, fontweight="bold", color='black', pad=20)

    plt.legend(
        wedges, labels,
        title="Emotions",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=11,
        title_fontsize=12,
        labelcolor='black',
        facecolor='white'
    )

    plt.tight_layout()
    output_path = get_output_path(unique_id, patient_id, "emotions_pie_darktheme_whitebg.png")
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path





# === 2. Timeline of Previous Hernia Repairs ===
# def plot_hernia_repairs(repairs):
#     if not repairs:
#         return
#     years = [int(r["Year"]) for r in repairs if "Year" in r]
#     types = [r["Type"] for r in repairs]
#     plt.figure(figsize=(10, 2))
#     plt.scatter(years, [1]*len(years), s=100, c="red", label="Repair")
#     for i, (y, t) in enumerate(zip(years, types)):
#         plt.text(y, 1.02, f"{t} ({y})", rotation=45, ha="right", va="bottom")
#     plt.yticks([])
#     plt.xlabel("Year")
#     plt.title("Previous Hernia Repairs Timeline")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, "hernia_timeline.png"))
#     plt.close()


def plot_hernia_repairs(repairs, unique_id, patient_id):
    if not repairs:
        return None

    years = [int(r["Year"]) for r in repairs if "Year" in r]
    types = [r.get("Type", "Unknown") for r in repairs]

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.vlines(years, ymin=0, ymax=1, color="skyblue", linewidth=2)
    ax.scatter(years, [1]*len(years), s=200, color="steelblue", zorder=3, edgecolors="white", linewidth=2)

    for y, t in zip(years, types):
        ax.text(y, 1.05, f"{t}\n({y})", ha="center", va="bottom", fontsize=10, fontweight="bold", rotation=0, color="#333")

    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_xticks(sorted(set(years)))
    ax.set_xlabel("Year", fontsize=12, weight="bold")
    ax.set_title("🩺 Timeline of Previous Hernia Repairs", fontsize=14, weight="bold", pad=20)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    output_path = get_output_path(unique_id, patient_id, "hernia_timeline.png")
    plt.savefig(output_path)
    plt.close()
    return output_path



# plot_hernia_repairs(data["metadata"]["Previous Hernia Repairs"])

# === 3. Word Cloud from QoL Summary ===
def plot_wordcloud(qol_summary, unique_id, patient_id):
    if isinstance(qol_summary, str):
        full_text = qol_summary
    elif isinstance(qol_summary, dict):
        full_text = " ".join(qol_summary.values())
    else:
        print("Invalid QoL summary format for wordcloud.")
        return None

    wc = WordCloud(width=800, height=400, background_color="white",
                   stopwords=set(stopwords.words("english"))).generate(full_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud - QoL Summary")
    plt.tight_layout()

    output_path = get_output_path(unique_id, patient_id, "qol_wordcloud.png")
    plt.savefig(output_path)
    plt.close()
    return output_path


# plot_wordcloud(data["qol_summary"])