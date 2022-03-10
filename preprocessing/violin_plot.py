import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataframe = pd.read_csv("HEF.csv", error_bad_lines=False, encoding="ISO-8859-1")

dataframe = dataframe[dataframe['HFE'] < 50]
print(dataframe.info())
print(dataframe.describe())
population = dataframe.HFE

sns.set_theme(style="whitegrid")

# tips = sns.load_dataset("tips")

ax = sns.violinplot(x=population)

# dataframe = pd.read_csv("HEF.csv", error_bad_lines=False, encoding="ISO-8859-1")
# print(dataframe.head())
# print(dataframe.isnull().values.any())

# import pandas as pd
# import matplotlib.pyplot as plt

# dataframe = pd.read_csv("HEF.csv", error_bad_lines=False, encoding="ISO-8859-1")

# population = dataframe.HFE
# # life_exp = dataframe.life_exp
# # gdp_cap = dataframe.gdp_cap

# # Extract Figure and Axes instance
# fig, ax = plt.subplots()

# Create a plot
# ax.violinplot([population])

# # Add title
# ax.set_title('Violin Plot')
plt.show()

dataframe = dataframe.groupby("country").last()
dataframe = dataframe.sort_values(by=["population"], ascending=False)
dataframe = dataframe.iloc[10:]
print(dataframe)

# Create figure with three axes
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

# Plot violin plot on axes 1
ax1.violinplot(dataframe.population, showmedians=True)
ax1.set_title('Population')

# Plot violin plot on axes 2
ax2.violinplot(life_exp, showmedians=True)
ax2.set_title('Life Expectancy')

# Plot violin plot on axes 3
ax3.violinplot(gdp_cap, showmedians=True)
ax3.set_title('GDP Per Cap')

plt.show()


fig, ax = plt.subplots()
ax.violinplot(gdp_cap, showmedians=True)
ax.set_title('violin plot')
ax.set_xticks([1])
ax.set_xticklabels(["Country GDP",])
plt.show()



fig, ax = plt.subplots()
ax.violinplot(gdp_cap, showmedians=True, vert=False)
ax.set_title('violin plot')
ax.set_yticks([1])
ax.set_yticklabels(["Country GDP",])
ax.tick_params(axis='y', labelrotation = 90)
plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
ax1.violinplot(population, showmedians=True, showmeans=True, vert=False)
ax1.set_title('Population')

ax2.violinplot(life_exp, showmedians=True, showmeans=True, vert=False)
ax2.set_title('Life Expectancy')

ax3.violinplot(gdp_cap, showmedians=True, showmeans=True, vert=False)
ax3.set_title('GDP Per Cap')
plt.show()