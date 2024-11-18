import json
from tqdm import tqdm
from collections import Counter

# Load the JSON file
with open('data/relationships.json', 'r') as file:
    data = json.load(file)

# Load relational aliases
alias_map = {}
with open('data/relational_aliases.txt', 'r') as alias_file:
    for line in alias_file:
        aliases = line.strip().split(',')
        primary_predicate = aliases[0].strip()
        for alias in aliases:
            alias_map[alias.strip()] = primary_predicate

# Initialize counters
predicate_counter = Counter()
subject_counter = Counter()
object_counter = Counter()

# Process each entry in the JSON data to count occurrences
for entry in tqdm(data):
    for relationship in entry.get('relationships', []):
        # Map predicate to primary predicate using alias_map
        predicate = relationship.get("predicate", "").lower()
        primary_predicate = alias_map.get(predicate, predicate)
        
        # Count predicates
        predicate_counter[primary_predicate] += 1

        # Extract subject and object names, handling both "names" (list) and "name" (string) cases
        subject_names = relationship.get("subject", {}).get("names") or [relationship.get("subject", {}).get("name")]
        object_names = relationship.get("object", {}).get("names") or [relationship.get("object", {}).get("name")]

        # Count each name in the subject and object lists, ignoring None values
        for name in filter(None, subject_names):
            subject_counter[name.lower()] += 1
        for name in filter(None, object_names):
            object_counter[name.lower()] += 1

# Filter predicates with more than 1000 counts
frequent_predicates = {pred for pred, count in predicate_counter.items() if count > 10000}

# Generate prompts and labels for filtered predicates
output_data = []
for entry in tqdm(data):
    for relationship in entry.get('relationships', []):
        # Map predicate to primary predicate using alias_map
        predicate = relationship.get("predicate", "").lower()
        primary_predicate = alias_map.get(predicate, predicate)
        
        if primary_predicate in frequent_predicates:
            # Extract subject and object names, handling both "names" (list) and "name" (string) cases
            subject_name = (relationship.get("subject", {}).get("names") or 
                            [relationship.get("subject", {}).get("name")])[0]
            object_name = (relationship.get("object", {}).get("names") or 
                           [relationship.get("object", {}).get("name")])[0]

            # Only proceed if both subject_name and object_name are valid (not None)
            if subject_name and object_name:
                # Format prompt and label
                subject_name, object_name = subject_name.lower(), object_name.lower()
                prompt = f"a {subject_name} {predicate} a {object_name}"
                label = f"{primary_predicate}"
                output_data.append({"prompt": prompt, "label": label})

# Get the top 10 items by count
top_predicates = predicate_counter.most_common(30) 
top_subjects = subject_counter.most_common(10)
top_objects = object_counter.most_common(10)

# Print results
print("Top 30 Predicates:", top_predicates)
print("Top 10 Subjects:", top_subjects)
print("Top 10 Objects:", top_objects)


# Save the output to a JSON file
with open('data/relation_prediction.json', 'w') as outfile:
    json.dump(output_data, outfile, indent=4)

print("Filtered data saved to 'relation_prediction.json'")
