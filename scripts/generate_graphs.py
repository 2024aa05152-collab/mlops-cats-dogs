#!/usr/bin/env python3
"""
Generate monitoring graphs for M5 presentation.
Creates static PNG charts from collected metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Output directory - save to images folder
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample data (in production, fetch from API)
data = {
    "total_predictions": 150,
    "labeled_predictions": 50,
    "accuracy": 0.82,
    "label_distribution": {"cat": 78, "dog": 72},
    "confidences": np.random.uniform(0.5, 0.95, 30).tolist(),
    "latencies": np.random.normal(0.2, 0.05, 30).tolist()
}

def create_label_distribution_chart():
    """Pie chart for label distribution."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = list(data["label_distribution"].keys())
    sizes = list(data["label_distribution"].values())
    colors = ['#FF6B6B', '#4ECDC4']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        explode=explode,
        labels=labels, 
        colors=colors,
        autopct='%1.1f%%',
        shadow=True, 
        startangle=90,
        textprops={'fontsize': 14}
    )
    
    ax.set_title('Prediction Label Distribution\n(Cats vs Dogs)', fontsize=16, fontweight='bold')
    
    # Add legend
    ax.legend(wedges, [f'{l}: {s} predictions' for l, s in zip(labels, sizes)],
              title="Labels", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/label_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {OUTPUT_DIR}/label_distribution.png")

def create_accuracy_chart():
    """Bar chart for accuracy metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Overall\nAccuracy', 'Cat\nAccuracy', 'Dog\nAccuracy']
    accuracies = [0.82, 0.85, 0.79]
    colors = ['#667eea', '#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(categories, accuracies, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracy Metrics', fontsize=16, fontweight='bold')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/accuracy_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {OUTPUT_DIR}/accuracy_metrics.png")

def create_confidence_chart():
    """Line chart for prediction confidence over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    confidences = sorted(data["confidences"])
    x = range(len(confidences))
    
    ax.plot(x, confidences, marker='o', linewidth=2, markersize=6, 
            color='#667eea', label='Confidence')
    ax.fill_between(x, confidences, alpha=0.3, color='#667eea')
    
    # Add threshold line
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold (0.5)')
    
    ax.set_xlabel('Request Number', fontsize=12)
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title('Prediction Confidence Distribution', fontsize=16, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add stats annotation
    mean_conf = np.mean(confidences)
    ax.text(0.02, 0.95, f'Mean: {mean_conf:.3f}', transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confidence_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {OUTPUT_DIR}/confidence_distribution.png")

def create_latency_chart():
    """Histogram for latency distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    latencies = [l for l in data["latencies"] if l > 0]
    
    # Create histogram
    n, bins, patches = ax.hist(latencies, bins=15, color='#4ECDC4', 
                                edgecolor='white', linewidth=1.5)
    
    # Color bars based on latency
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0.15:
            patch.set_facecolor('#4ECDC4')  # Green - fast
        elif left_edge < 0.25:
            patch.set_facecolor('#FFE66D')  # Yellow - moderate
        else:
            patch.set_facecolor('#FF6B6B')  # Red - slow
    
    ax.set_xlabel('Latency (seconds)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Request Latency Distribution', fontsize=16, fontweight='bold')
    
    # Add stats
    mean_lat = np.mean(latencies)
    ax.axvline(x=mean_lat, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_lat:.3f}s')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/latency_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {OUTPUT_DIR}/latency_distribution.png")

def create_request_count_chart():
    """Stacked bar chart for request types."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    endpoints = ['/predict', '/predict_with_label', '/health', '/metrics']
    cat_preds = [45, 25, 0, 0]
    dog_preds = [52, 28, 0, 0]
    
    x = np.arange(len(endpoints))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cat_preds, width, label='Cat Predictions', color='#FF6B6B')
    bars2 = ax.bar(x + width/2, dog_preds, width, label='Dog Predictions', color='#4ECDC4')
    
    ax.set_xlabel('API Endpoint', fontsize=12)
    ax.set_ylabel('Request Count', fontsize=12)
    ax.set_title('API Request Breakdown by Endpoint', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(endpoints)
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/request_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {OUTPUT_DIR}/request_breakdown.png")

def create_summary_dashboard():
    """Create a summary dashboard with all metrics."""
    fig = plt.figure(figsize=(16, 12))
    
    # Title
    fig.suptitle('M5: Monitoring Dashboard - Cats vs Dogs Classification', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # 1. Accuracy gauge (simplified as bar)
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Accuracy']
    values = [data["accuracy"]]
    colors = ['#667eea']
    ax1.barh(categories, values, color=colors)
    ax1.set_xlim(0, 1)
    ax1.set_title('Overall Accuracy', fontsize=14, fontweight='bold')
    ax1.text(values[0]/2, 0, f'{values[0]*100:.1f}%', ha='center', va='center', 
             fontsize=20, fontweight='bold', color='white')
    
    # 2. Total predictions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.pie([data["label_distribution"]["cat"], data["label_distribution"]["dog"]], 
            labels=['Cat', 'Dog'], colors=['#FF6B6B', '#4ECDC4'],
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Label Distribution', fontsize=14, fontweight='bold')
    
    # 3. Confidence distribution
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(data["confidences"], marker='o', linewidth=2, color='#667eea')
    ax3.fill_between(range(len(data["confidences"])), data["confidences"], alpha=0.3, color='#667eea')
    ax3.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
    ax3.set_title('Prediction Confidence Over Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Request')
    ax3.set_ylabel('Confidence')
    ax3.legend()
    
    # 4. Latency histogram
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(data["latencies"], bins=10, color='#4ECDC4', edgecolor='white')
    ax4.set_title('Latency Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Seconds')
    ax4.set_ylabel('Count')
    
    # 5. Metrics summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    metrics_text = f"""
    📊 Summary Metrics
    ═══════════════════
    
    Total Predictions: {data["total_predictions"]}
    With True Labels: {data["labeled_predictions"]}
    Overall Accuracy:  {data["accuracy"]*100:.1f}%
    
    Cat Predictions:   {data["label_distribution"]["cat"]}
    Dog Predictions:   {data["label_distribution"]["dog"]}
    
    Avg Confidence:    {np.mean(data["confidences"]):.3f}
    Avg Latency:       {np.mean(data["latencies"]):.3f}s
    """
    ax5.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
             verticalalignment='center', transform=ax5.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.savefig(f'{OUTPUT_DIR}/summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {OUTPUT_DIR}/summary_dashboard.png")

def main():
    print("\n🎨 Generating M5 Monitoring Graphs...")
    print("=" * 50)
    
    create_label_distribution_chart()
    create_accuracy_chart()
    create_confidence_chart()
    create_latency_chart()
    create_request_count_chart()
    create_summary_dashboard()
    
    print("=" * 50)
    print(f"\n📁 All graphs saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")

if __name__ == "__main__":
    main()

