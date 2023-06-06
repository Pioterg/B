import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from  PIL import Image

#zad1
# x = np.arange(3, 7.25, 0.25)
# y = np.cos(x)/x**2
# plt.plot(x, y, label='f(x) = cos(x)/x^2')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.xlim(3,7)
# plt.title('Wykres funkcji f(x)')
# plt.legend()
# print('Wektor x:', x)
# print('Wektor y:', y)
# plt.savefig('Piotr_Sitkowski_zad1.png')
# plt.show()

#zad2
# x1 = np.arange(-4, 4.1, 0.1)
# x2 = np.arange(-4, 4.1, 0.1)
# y1 = 5*x1**2-3*x1+2
# y2 = -2*x2**3+5
#
# plt.subplot(3, 2, 4)
# plt.plot(x1, y1, 'r', label='5x^2-3x+2')
# plt.title('pierwszy wykres')
# plt.xlabel('x')
# plt.ylabel('wynik funkcji')
# plt.xlim([-4, 4])
# plt.legend()
#
# plt.subplot(3, 2, 5)
# plt.plot(x2, y2, 'go', linewidth=5, label='-2x^3+5')
# plt.title('Drugi wykres')
# plt.xlabel('x')
# plt.ylabel('wynik funkcji')
# plt.xlim([-4, 4])
# plt.legend(loc=9)
#
# plt.subplots_adjust(hspace=0.5)
# plt.savefig('Piotr_Sitkowski_zad2.png')
# plt.show()

#zad3
# df = pd.read_csv('wine.data', header=None, skiprows=1)
# print(df)
# losowe = df.sample(n=100, replace=True)
# print(losowe)
# grupowana = df.groupby(0).size()
# print(grupowana)
# grupowana.plot(kind='pie', autopct='%.2f%%')
# plt.title('Udział klas')
# plt.legend()
# plt.savefig('Piotr_Sitkowski_zad3.png')
# plt.show()

#zad4
df = pd.read_csv('wine.data', header=None, skiprows=1)
print(df)
# df.columns = ['Class', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium', 'Total Phenols',
#               'Flavanoids', 'Nonflavanoid Phenols', 'Proanthocyanins', 'Color Intensity', 'Hue',
#               'OD280/OD315 of Diluted Wines', 'Proline']
df[1] = pd.to_numeric(df[1], errors='coerce')
sns.set(style='whitegrid')  # Ustawienie stylu wykresu na podstawowy
sns.barplot(x=0, y=1, data=df, ci=None)
plt.xlabel('Klasa')
plt.ylabel('Średnia wartość alkoholu')
plt.title('Średnie wartości alkoholu dla każdej klasy')
plt.savefig('Piotr_Sitkowski_zad4.png')
plt.show()