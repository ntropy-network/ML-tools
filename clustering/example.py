import pandas as pd
import clustering

sentences = [
    'how much for teeth implants',
    'how much sugar should i eat a day',
    'what is the price of a stamp',
    'what do us postage stamps cost',
    'recommended limit of sugar per day',
    'how much is us stamp',
    'about how much do implants cost tooth',
    'how long cooked eggs in fridge',
    'how long is an unrefrigerated boiled egg good',
    'recommended added sugar intake per day',
    'how much will dentures cost',
    'how long do do eggs last in a refrigerator',
    'usa postal stamp cost',
    'how long does hard boiled eggs stay good',
    'what does it cost for dental implants',
    'how much sugar should you have in one day',
]

df = pd.DataFrame(data=sentences, columns=['text'])
clusters, df_clustered = clustering.cluster(df, text_key='text', min_cluster_size=3, threshold=0.67)

# Print all clusters seperated by two newlines
for _, cluster in df_clustered.groupby('cluster_id'):
    print('\n'.join(cluster['text']))
    print('')
