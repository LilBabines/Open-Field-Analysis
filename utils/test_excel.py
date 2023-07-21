import read_exel

df=read_exel.get_df()
time=df.iloc[1]['end_time']
print(time.minute+time.hour*60)