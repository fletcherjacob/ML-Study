import pandas as pd

X = pd.read_csv(r"/home/fletcher/Repos/ML-Study/adult.data")
print(X.head())

#if header in the dataset it must be specified or the first row will be listed as
#as the header
X = pd.read_csv(r"/home/fletcher/Repos/ML-Study/adult.data", header=None)
print(X.head())



#Sample
df = pd.DataFrame([ ('John',151.0),
                    ('Jerry', 205),
                    ('Steve', 186)],
            columns=('Name','Amount'))
print(df)


 #converts column hot-encode, can be specified for multiples
df2 = pd.get_dummies(df, columns =['Name'])
print(df2)


print(df['Name'].unique()) #displays each category
print(len(df['Name'].unique())) #displays count of unique categories
