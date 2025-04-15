import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv('chatbot_evaluation_results.csv')

# Function to extract numeric scores from evaluation text
def parse_scores(evaluation_text):
    criteria = ['Response correctness', 'Quality of feedback', 'Relevance',
                'Educational effectiveness', 'Engagement potential']
    scores = {}
    for criterion in criteria:
        pattern = rf'{criterion}:\s*(\d|N/A)'
        match = re.search(pattern, evaluation_text)
        if match:
            score = match.group(1)
            scores[criterion] = int(score) if score.isdigit() else None
    return scores

# Apply parsing function to extract structured scores
df_scores = df['evaluation'].apply(parse_scores).apply(pd.Series)

# Combine extracted scores with original dataframe
df_final = pd.concat([df, df_scores], axis=1)

# Calculate average scores per response type
avg_scores = df_final.groupby('response_type').mean(numeric_only=True)
print("Average Scores per Response Type:")
print(avg_scores)

# Save average scores to CSV
avg_scores.to_csv('aggregated_chatbot_scores.csv')

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_scores.reset_index().melt(id_vars='response_type'),
            x='response_type', y='value', hue='variable')

plt.title('Average Evaluation Scores by Response Type')
plt.xlabel('Response Type')
plt.ylabel('Average Score')
plt.ylim(0, 5)
plt.legend(title='Criteria', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('evaluation_scores.png')
plt.show()
