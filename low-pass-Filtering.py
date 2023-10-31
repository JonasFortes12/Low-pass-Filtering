import numpy as np
import math as mt
import matplotlib.pyplot as plt

# Função para plotagem de gráficos
def plot_graph(x, y, title, xlabel, ylabel, label='', color='blue' ):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=label, color=color)
    # Adicionar um título ao gráfico
    plt.title(title)
    # Adicionar rótulos aos eixos
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    # Adicionar uma legenda
    plt.legend()
    
    plt.show()

#__________________________ Questão 01 _____________________________________

N = 100  # Número de amostras
n = np.arange(N)  # Vetor de amostras de 0 a N-1

# Frequências desejadas em radianos por amostra
frequencies = [0.1 * mt.pi, 0.6 * mt.pi, 0.9 * mt.pi]

# Gera o sinal somando as componentes de frequência
Xn = np.sum([np.cos(omega * n) for omega in frequencies], axis=0)

plot_graph(n, Xn, 'Sinal de entrada X[n]', 'Tempo', 'Amplitude')

#__________________________ Questão 02 _____________________________________

# Calcula o módulo da transformada de Fourier do sinal X[n]
XnFT = np.abs(np.fft.fftshift(np.fft.fft(Xn)))

# Calcula as frequências correspondentes às amostras
freqs = np.fft.fftfreq(N)

plot_graph(freqs, XnFT, 'Módulo da FT do sinal X[n]', 'Frequência', 'Amplitude', color='black')

#__________________________ Questão 03 _____________________________________


# Parâmetros do filtro passa-baixa ideal
cutOffFreq = 0.75 * np.pi # Frequência de corte
a = 15  # Atraso
# Tamanho da resposta ao impulso
M = 2 * a  # M = 30

# Amostras de 0 até M-1
n = np.arange(M)


# Resposta ao impulso do filtro passa-baixa FIR com janela retangular
# Normalização por (1 / pi) é uma convenção padrão
h_ideal = (1 / np.pi) * np.sin(cutOffFreq * (n - a)) / (n - a)
h_ideal[a] = cutOffFreq / np.pi  # Lidando com a divisão por zero


plot_graph(n, h_ideal, 'Resposta ao Impulso do Filtro Passa-Baixa FIR h[n]', 'Tempo', 'Amplitude')


#__________________________ Questão 04 _____________________________________

# Calcula o módulo da resposta em frequência do sistema (FT da reposta ao impulso h[n])
h_idealFT = np.abs(np.fft.fftshift(np.fft.fft(h_ideal)))

# Calcula as frequências correspondentes às amostras
freqs = np.fft.fftfreq(M)

plot_graph(freqs, h_idealFT, 'Módulo da Resposta em Frequência do Sistema', 'Frequência', 'Amplitude', color='black')