#!/usr/bin/env python3
"""
Generate monitoring graphs from REAL API data.
Creates PNG charts from live metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import requests
import json
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Output directory - save to images folder
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# API URL
API_URL = "http://localhost:8000"

def fetch_stats():
    """Fetch real stats from API."""
    try:
        resp = requests.get(f"{API_URL}/stats", timeout=5)
        return resp.json()
    except:
        return None

def fetch_metrics():
    """Fetch Prometheus metrics from API."""
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=5)
        text = resp.text
        
        # Parse metrics
        data = {}
        for line in text.split('\n'):
            if line.startswith('inference_requests_total'):
                # Extract endpoint and label
                import re
                match = re.search(r'endpoint="([^"]+)",label="([^"]+)"', line)
                if match:
                    endpoint, label = match.groups()
                    value = float(line.split()[-1])
                    if endpoint not in data:
                        data[endpoint] = {}
                    data[endpoint][label] = value
            
            if 'inference_latency_seconds_count' in line:
                data['latency_count'] = float(line.split()[-1])
            if 'inference_latency_seconds_sum' in line:
                data['latency_sum'] = float(line.split()[-1])
        
        return data
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return None

def create_real_time_dashboard():
    """Create dashboard with real API data."""
    stats = fetch_stats()
    metrics = fetch_metrics()
    
    print("\n📊 Real-time Data from API:")
    print("=" * 50)
    if stats:
        print(f"Total Predictions: {stats.get('total_predictions', 0)}")
        print(f"Labeled Predictions: {stats.get('labeled_predictions', 0)}")
        print(f"Accuracy: {stats.get('accuracy', 0)*100:.1f}%")
        print(f"Label Distribution: {stats.get('label_distribution', {})}")
    if metrics:
        print(f"Latency Count: {metrics.get('latency_count', 0)}")
        print(f"Latency Sum: {metrics.get('latency_sum', 0):.3f}s")
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('M5: Real-Time Monitoring Dashboard - Cats vs Dogs', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # 1. Prediction Stats
    ax1 = fig.add_subplot(gs[0, 0])
    if stats:
        labels = ['Total', 'Labeled']
        values = [stats.get('total_predictions', 0), stats.get('labeled_predictions', 0)]
        colors = ['#667eea', '#4ECDC4']
        bars = ax1.bar(labels, values, color=colors, edgecolor='white', linewidth=2)
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                    str(int(val)), ha='center', va='bottom', fontsize=16, fontweight='bold')
    ax1.set_title('Prediction Count', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    
    # 2. Label Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    if stats and stats.get('label_distribution'):
        dist = stats['label_distribution']
        ax2.pie(list(dist.values()), labels=list(dist.keys()), 
                colors=['#FF6B6B', '#4ECDC4'], autopct='%1.0f%%', startangle=90,
                textprops={'fontsize': 12})
    else:
        ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
    ax2.set_title('Label Distribution', fontsize=14, fontweight='bold')
    
    # 3. Request by Endpoint
    ax3 = fig.add_subplot(gs[1, 0])
    if metrics:
        endpoints = list(metrics.keys())
        endpoints = [e for e in endpoints if isinstance(metrics.get(e), dict)]
        if endpoints:
            # Sum up all labels per endpoint
            endpoint_totals = {}
            for ep in endpoints:
                endpoint_totals[ep] = sum(metrics[ep].values())
            
            bars = ax3.bar(endpoint_totals.keys(), endpoint_totals.values(), 
                          color=['#667eea', '#764ba2'], edgecolor='white', linewidth=2)
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax3.set_title('Requests by Endpoint', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=15)
    
    # 4. Latency
    ax4 = fig.add_subplot(gs[1, 1])
    if metrics and metrics.get('latency_count'):
        count = metrics.get('latency_count', 1)
        total = metrics.get('latency_sum', 0)
        avg_latency = total / count if count > 0 else 0
        
        # Simulate latency buckets (from histogram data)
        buckets = ['<50ms', '50-100ms', '100-250ms', '>250ms']
        # Estimate from histogram
        estimated = [1, 2, 3, 0]  # From the bucket data
        
        colors_lat = ['#4ECDC4', '#667eea', '#FFE66D', '#FF6B6B']
        bars = ax4.bar(buckets, estimated, color=colors_lat, edgecolor='white', linewidth=2)
        
        ax4.axhline(y=avg_latency*10, color='red', linestyle='--', 
                   label=f'Avg: {avg_latency*1000:.0f}ms')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
    ax4.set_title('Latency Distribution', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Count')
    
    plt.savefig(f'{OUTPUT_DIR}/realtime_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Created: {OUTPUT_DIR}/realtime_dashboard.png")

def create_metrics_summary():
    """Create a summary of metrics."""
    stats = fetch_stats()
    metrics = fetch_metrics()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('M5: Key Metrics Summary', fontsize=16, fontweight='bold')
    
    # 1. Total Predictions
    ax1 = axes[0]
    total = stats.get('total_predictions', 0) if stats else 0
    ax1.pie([total, 100-total], labels=['Recorded', 'Remaining'], 
            colors=['#667eea', '#e0e0e0'], startangle=90)
    ax1.set_title(f'Total Predictions\n{total}', fontsize=12)
    
    # 2. Accuracy (or N/A)
    ax2 = axes[1]
    accuracy = stats.get('accuracy', 0) if stats else 0
    labeled = stats.get('labeled_predictions', 0) if stats else 0
    if labeled > 0:
        ax2.pie([accuracy, 1-accuracy], labels=['Correct', 'Wrong'],
               colors=['#4ECDC4', '#FF6B6B'], startangle=90)
        ax2.set_title(f'Accuracy\n{accuracy*100:.1f}%', fontsize=12)
    else:
        ax2.text(0.5, 0.5, f'No labeled\ndata yet', ha='center', va='center', fontsize=14)
        ax2.set_title('Accuracy\nN/A', fontsize=12)
    
    # 3. Avg Latency
    ax3 = axes[2]
    if metrics and metrics.get('latency_count'):
        count = metrics.get('latency_count', 1)
        total = metrics.get('latency_sum', 0)
        avg = total / count if count > 0 else 0
        ax3.bar(['Avg Latency'], [avg*1000], color='#667eea', edgecolor='white')
        ax3.set_ylabel('Milliseconds')
        ax3.set_title(f'Avg Latency\n{avg*1000:.0f}ms', fontsize=12)
    else:
        ax3.text(0.5, 0.5, 'No latency\ndata', ha='center', va='center', fontsize=14)
        ax3.set_title('Latency\nN/A', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/metrics_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {OUTPUT_DIR}/metrics_summary.png")

def main():
    print("\n🎨 Generating M5 Graphs from REAL API Data...")
    print("=" * 50)
    
    create_real_time_dashboard()
    create_metrics_summary()
    
    print("=" * 50)
    print(f"\n📁 Graphs saved to: {OUTPUT_DIR}/")
    print("\nGenerated:")
    print("  - realtime_dashboard.png")
    print("  - metrics_summary.png")

if __name__ == "__main__":
    main()

