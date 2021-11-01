Description of Python files

1. construct_clusters_df.py: load the 200,000 users dataframe to construct clusters of different user type (from very active to rather inactive). Used for Fig. 2a

2. construct_cohorts_df.py: load the 200,000 users dataframe to sort users into cohorts with mutual starting year

3. intereventtimes_clusters_first_year.py: compute arrays of intereventtimes of the different user types for their first year of activity. Intereventtimes are the time periods between recorded posts.

4. intereventtimes_cohorts_first_year.py: same as before but for the cohorts

5. lifetime_dist_clusters.py: obtain dates for first post for the user types. This can be plotted as cumulative starting date distribution (as in the SI) or as in Fig. 2b for showing that more active user types tend to have joined the platform later
