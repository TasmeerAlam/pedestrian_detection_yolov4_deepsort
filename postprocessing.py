import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv('mass_10th.csv', index_col= None)

time = datetime(year = 2019, month = 5, day = 9, hour = 6, minute = 00, second = 29)

df['Time'] = df.Frame.apply(lambda x: time + timedelta(seconds=x/10))

print(df.head(120))

groupby_obj = df.groupby('Tracker ID')

residence_frame_limit = 5
direction_tracker = {}

for id in df["Tracker ID"].unique():
	marker_previous = None
	tracker = []

	count = 0

	for index, marker in df[df['Tracker ID'] == id].Marker.diff(-1).items():
		if count >= residence_frame_limit:
			tracker.append(df.loc[index]['Marker'])
			count = 0

		if marker == marker_previous:
			if len(tracker) > 0 and df.loc[index]['Marker'] == tracker[-1]:
				pass
			else:
				count +=1

		marker_previous = marker


	direction_tracker[id] = tracker

print(direction_tracker)