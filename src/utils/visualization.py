import matplotlib.pyplot as plt
from collections import Counter
from src.data.dataset import SingleLabelRelationalDataset, MultiLabelRelationalDataset
import numpy as np
import os
def plot_label_distribution(loader, n_display=10, title="Label Distribution", save_path=None):
    """
    Plots the label distribution from a DataLoader and includes additional plots
    for entity and relation distributions.

    Args:
        loader: A PyTorch DataLoader that provides batches of data and labels.
        n_display: The number of evenly spaced labels to display in the plot.
    """
    # Aggregate labels from the DataLoader
    all_labels = []
    for _, labels in loader:
        if isinstance(loader.dataset.dataset, SingleLabelRelationalDataset):
            labels_translated = [loader.dataset.dataset.idx_to_label[element.item()] for element in labels]
        elif isinstance(loader.dataset.dataset, MultiLabelRelationalDataset):
            labels_translated = loader.dataset.dataset.convert_to_labels(labels)
            labels_translated = [x for xs in labels_translated for x in xs]
        all_labels.extend(labels_translated)

    # Split labels into relations and entities if applicable
    relations = []
    entities = []
    for label in all_labels:
        if isinstance(label, str) and "_" in label:
            parts = label.split("_")
            if len(parts) >= 3:  # Ensure there are enough parts to separate relation and entities
                relations.append("_".join(parts[:-2]))  # Everything before last two as the relation
                entities.extend(parts[-2:])  # Last two parts as entities

    # Count occurrences of each label
    label_counts = Counter(all_labels)
    relation_counts = Counter(relations) if relations else {}
    entity_counts = Counter(entities) if entities else {}

    # Sort labels and counts lexicographically
    sorted_labels_and_counts = sorted(label_counts.items())
    all_labels, all_counts = zip(*sorted_labels_and_counts)

    # Determine whether to show all ticks or evenly spaced ones
    if len(all_labels) <= 20:
        x_tick_labels = list(all_labels)  # Show all labels
    else:
        indices = np.linspace(0, len(all_labels) - 1, n_display, dtype=int)  # Evenly spaced indices
        x_tick_labels = ["" if i not in indices else all_labels[i] for i in range(len(all_labels))]

    # Create x-ticks
    x_ticks = list(range(len(all_labels)))  # X-axis positions for all bars

    # Determine subplot layout
    num_plots = 1 + bool(relations) + bool(entities)  # Count the required subplots
    figsize = (14, 6 * num_plots)

    # Plot the label distribution
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16)

    # Plot 1: Label distribution
    plt.subplot(num_plots, 1, 1)
    plt.bar(x_ticks, all_counts, color='skyblue')
    plt.xticks(x_ticks, x_tick_labels, rotation=45, ha="right")  # Add custom x-ticks
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.title("Label Distribution")

    # Plot 2: Relation distribution
    if relations:
        sorted_relations_and_counts = sorted(relation_counts.items())
        relation_labels, relation_counts_values = zip(*sorted_relations_and_counts)
        plt.subplot(num_plots, 1, 2)
        plt.bar(relation_labels, relation_counts_values, color='lightcoral')
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Relations")
        plt.ylabel("Misclassifications")
        plt.title("Relation Distribution")

    # Plot 3: Entity distribution
    if entities:
        sorted_entities_and_counts = sorted(entity_counts.items())
        entity_labels, entity_counts_values = zip(*sorted_entities_and_counts)
        plt.subplot(num_plots, 1, 3)
        plt.bar(entity_labels, entity_counts_values, color='lightgreen')
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Entities")
        plt.ylabel("Misclassifications")
        plt.title("Entity Distribution")

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    if save_path:
        filename = os.path.join(save_path, '_'.join(title.split(' ')) + "_label_distribution.png")
        plt.savefig(filename)
    plt.close()

def plot_train_val_metrics(train_losses, val_losses, train_accuracies, val_accuracies, checkpoint_dir):
    """
    Plots training/validation losses and accuracies for each probe.
    """
    plt.figure(figsize=(14, 10))

    # Plot train/validation losses
    plt.subplot(2, 1, 1)
    for probe_name in train_losses:
        plt.plot(train_losses[probe_name], label=f"{probe_name} Train Loss")
        plt.plot(val_losses[probe_name], label=f"{probe_name} Val Loss", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()

    # Plot train/validation accuracies
    plt.subplot(2, 1, 2)
    for probe_name in train_accuracies:
        plt.plot(train_accuracies[probe_name], label=f"{probe_name} Train Accuracy")
        plt.plot(val_accuracies[probe_name], label=f"{probe_name} Val Accuracy", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir,"train_val_metrics_with_accuracy.png"))
    plt.close()

def plot_misclassification_distribution(probe_name: str, accuracy: float, misclassified_labels: Counter, output_dir: str):
    """
    Plots the distribution of misclassified labels, focusing only on misclassified instances.
    
    Args:
        probe_name: Name of the probe being analyzed.
        accuracy: Accuracy of the probe on the test set.
        misclassified_labels: Counter object with misclassified labels and their counts.
        output_dir: Directory where the output plots will be saved.
    """
    # Sort labels lexicographically
    sorted_labels_and_counts = sorted(misclassified_labels.items())
    labels, counts = zip(*sorted_labels_and_counts) if sorted_labels_and_counts else ([], [])

    # Plot the label distribution
    plt.figure(figsize=(14, 8))
    plt.bar(labels, counts, color='skyblue')
    plt.title(f"{probe_name} - Misclassification Distribution (Accuracy: {accuracy:.2f}%)", fontsize=16)
    plt.xlabel("Labels")
    plt.ylabel("Misclassifications")
    plt.xticks(rotation=45, ha="right")

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    filename = os.path.join(output_dir, f"{probe_name}_misclassification_distribution.png")
    plt.savefig(filename)
    plt.close()



def plot_top_misclassifications(probe_name: str, accuracy: float, misclassified_labels: Counter, reverse_misclassified_labels: Counter, swapped_relation_labels: Counter, swapped_relation_and_nouns_labels: Counter, output_dir: str):
    # Accumulate misclassification counts per noun and relation
    noun_counts = Counter()
    relation_counts = Counter()

    for label, count in misclassified_labels.items():
        parts = label.split('_')
        noun1 = parts[-2]
        noun2 = parts[-1]
        relation = '_'.join(parts[0:-2])
        relation_counts[relation] += count
        noun_counts[noun1] += count
        noun_counts[noun2] += count

    # Calculate total misclassifications and classify remaining as "Other"
    total_misclassifications = sum(misclassified_labels.values())
    reverse_total = sum(reverse_misclassified_labels.values())
    swapped_relation_total = sum(swapped_relation_labels.values())
    swapped_relation_and_nouns_total = sum(swapped_relation_and_nouns_labels.values())
    other_total = total_misclassifications - (reverse_total + swapped_relation_total + swapped_relation_and_nouns_total)

    pie_sizes = [reverse_total, swapped_relation_total, swapped_relation_and_nouns_total, other_total]
    pie_labels = ['Reverse Predictions', 'Relation Swapped', 'Relation and Entities Swapped', 'Other']

    # Set up figure with two columns: left for bar charts and right for a larger pie chart spanning two rows
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"{probe_name} - Test Accuracy: {accuracy:.2f}%", fontsize=16)

    # Plot noun misclassification counts (top left)
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    nouns, noun_miscounts = zip(*noun_counts.most_common())
    ax1.bar(nouns, noun_miscounts)
    ax1.set_xlabel("Entities")
    ax1.set_ylabel("Misclassifications")
    ax1.set_title("Misclassification Counts by Noun")
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot relation misclassification counts (bottom left)
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    relations, relation_miscounts = zip(*relation_counts.most_common())
    ax2.bar(relations, relation_miscounts)
    ax2.set_xlabel("Relations")
    ax2.set_ylabel("Misclassifications")
    ax2.set_title("Misclassification Counts by Relation")
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot pie chart of misclassification types (right side spanning both rows)
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax3.pie(pie_sizes, labels=pie_labels, autopct=lambda p: f'{p:.1f}%\n({int(p * total_misclassifications / 100)})')
    ax3.set_title("Misclassification Types")

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.savefig(os.path.join(output_dir,f"{probe_name}_misclassifications_by_entity_relation_and_type.png"))
    plt.close()
    