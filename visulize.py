import re
import os   
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

# === Create Output Folder ===
output_folder = "report_images"
os.makedirs(output_folder, exist_ok=True)

def get_output_path(unique_id, patient_id, filename):
    subfolder = f"{unique_id}_{patient_id.replace(' ', '_')}"
    folder_path = os.path.join("report_images", subfolder)
    os.makedirs(folder_path, exist_ok=True)
    return os.path.join(folder_path, filename)

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

def plot_wordcloud(qol_summary):
    if isinstance(qol_summary, str):
        full_text = qol_summary
    elif isinstance(qol_summary, dict):
        full_text = " ".join(qol_summary.values())
    else:
        print("Invalid QoL summary format for wordcloud.")
        return None

    # Tokenize and clean text
    words = re.findall(r'\b\w+\b', full_text.lower())
    filtered_words = [word for word in words if word not in stopwords.words('english')]

    # Count frequency
    word_freq = Counter(filtered_words)
    most_common_words = word_freq.most_common(15)

    # Generate word cloud
    wc = WordCloud(width=800, height=400, background_color="white",
                   stopwords=set(stopwords.words("english"))).generate(full_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud - QoL Summary")
    plt.tight_layout()

    # output_path = get_output_path(unique_id, patient_id, "qol_wordcloud.png")
    # plt.savefig(output_path)
    # plt.close()

    return {
        "top_words": most_common_words
    }
