import re
from collections import Counter
import networkx as nx
from wordcloud import WordCloud
from matplotlib.figure import Figure


# Note: We do NOT use 'import matplotlib.pyplot as plt' here.
# Using pyplot in a loop with Tkinter causes memory leaks.
# We use the object-oriented 'Figure' approach instead.

def create_wordcloud_fig(text_data):
    if not text_data: return None

    # Clean text (remove special characters)
    clean = re.sub(r'[^\w\s]', '', text_data)

    # Generate Cloud
    try:
        wc = WordCloud(width=400, height=300, background_color='#2b2b2b',
                       colormap='Blues', max_words=50).generate(clean)
    except ValueError:
        return None  # Handle empty text errors

    # Create a pure Figure object (no popup window)
    fig = Figure(figsize=(4, 3), dpi=100, facecolor='#2b2b2b')
    ax = fig.add_subplot(111)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def create_barchart_fig(text_data):
    if not text_data: return None

    words = re.findall(r'\w+', text_data.lower())
    stop_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on',
                  'with', 'as', 'by', 'are', 'it', 'or', 'this', 'that', 'be', 'from'}

    filtered = [w for w in words if w not in stop_words and len(w) > 3]
    counts = Counter(filtered).most_common(5)

    if not counts: return None

    labels, values = zip(*counts)

    fig = Figure(figsize=(4, 3), dpi=100, facecolor='#2b2b2b')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#2b2b2b')

    # Draw horizontal bars
    bars = ax.barh(labels, values, color='#00d4ff')

    # Styling for Dark Mode
    ax.tick_params(axis='x', colors='white', labelsize=8)
    ax.tick_params(axis='y', colors='white', labelsize=8)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('white')
    ax.set_title("Top Keywords", color='white', fontsize=10)
    fig.tight_layout()
    return fig


def create_knowledge_graph_fig(query, text_data):
    if not text_data: return None

    words = re.findall(r'\w+', text_data.lower())
    stop_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'it', 'that', 'for', 'with'}
    filtered = [w for w in words if w not in stop_words and len(w) > 4]

    # Get top 6 concepts
    top_concepts = [w[0] for w in Counter(filtered).most_common(6)]
    if not top_concepts: return None

    G = nx.Graph()
    center_node = "Query"
    G.add_node(center_node, color='#ff4444')  # Red for user query

    for concept in top_concepts:
        G.add_node(concept, color='#00C851')  # Green for found concepts
        G.add_edge(center_node, concept)

    fig = Figure(figsize=(4, 3), dpi=100, facecolor='#2b2b2b')
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Calculate layout
    pos = nx.spring_layout(G, seed=42)
    colors = [G.nodes[n].get('color', 'skyblue') for n in G.nodes]

    # Draw
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=500)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='white', alpha=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='white', font_weight='bold')

    fig.tight_layout()
    return fig