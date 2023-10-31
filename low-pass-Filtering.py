import numpy as np
import math as mt
import matplotlib.pyplot as plt
from scipy import signal

# Função para plotagem de gráficos
def plot_graph(x, y,title, xlabel, ylabel, y2=None, label='',label2='', color='blue', color2='red'):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=label, color=color)
    if(y2 is not None):
        plt.plot(x, y2, label=label2, color=color2)
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

# Definir valores do eixo X
x =  np.linspace(-np.pi, np.pi, N)

plot_graph(x, XnFT, 'Módulo da FT do sinal X[n]', 'Frequência', 'Amplitude', color='black')

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

# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, M)

plot_graph(x, h_idealFT, 'Módulo da Resposta em Frequência do Sistema', 'Frequência', 'Amplitude', color='black')

#__________________________ Questão 05 _____________________________________

# Calcula a resposta em fase da resposta em frequência do sistema
phase_h_idealFT = np.angle(h_idealFT)

plot_graph(x, phase_h_idealFT, 'Resposta em Fase do Sistema', 'Frequência', 'Amplitude', color='green')


#__________________________ Questão 06 _____________________________________

# Calcula o atraso de grupo do sistema 
frequencies, groupDelay = signal.group_delay((h_ideal, 1))

# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, len(groupDelay))

plot_graph(x, groupDelay, 'Atraso de Grupo do Sistema', 'Frequência (rad/sample)', 'Atraso de Grupo (samples)')


#__________________________ Questão 07 _____________________________________

# Realiza a convolução discreta para filtrar o sinal x[n] com a resposta ao impulso h[n]
Yn = np.convolve(Xn, h_ideal, mode='same') 
n = np.arange(len(Yn))

plot_graph(n, Yn,  'Sinal de Saída y[n] ( x[n]*h[n] )', 'Tempo', 'Amplitude')


#__________________________ Questão 08 _____________________________________

# Calcula o módulo da transformada de Fourier do sinal de saída y[n]
YnFT = np.abs(np.fft.fftshift(np.fft.fft(Yn)))

# Definir valores do eixo X
x =  np.linspace(-np.pi, np.pi, len(YnFT))

plot_graph(x, YnFT, 'Módulo da FT do sinal Y[n]', 'Frequência', 'Amplitude', color='black')

#__________________________ Questão 09 _____________________________________

# Gerando o sinal g[n]

N = 100  # Número de amostras
n = np.arange(N)  # Vetor de amostras de 0 a N-1

# Frequências desejadas em radianos por amostra
frequencies = [0.1 * mt.pi, 0.6 * mt.pi]

# Gera o sinal somando as componentes de frequência
Gn = np.sum([np.cos(omega * n) for omega in frequencies], axis=0)

plot_graph(n, Gn, 'Sinais G[n] e Y[n-a]', 'Tempo', 'Amplitude', Yn, 'Sinal g[n]', 'Sinal y[n-a]')

#__________________________ Questão 10 _____________________________________

# Explicar aqui quais as vantagens e desvantagens 
# deste filtro FIR a partir dos gráficos gerados até o momento



#__________________________ Questão 11 _____________________________________

# Frequência de corte desejada por amostra
cutOffFreq = 0.75

# Ordem do filtro Butterworth
order = 10

# Projetar o filtro Butterworth
# Os coeficientes ak e bk estão armazenados nos vetores a e b 
b, a = signal.butter(order, cutOffFreq, btype='low')

#__________________________ Questão 12 _____________________________________

# Calcula a resposta em frequência do filtro Butterworth 
w, hFreqResponse = signal.freqz(b, a)

# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, len(hFreqResponse))

plot_graph(x, hFreqResponse, 'Módulo da Resposta em Frequência do Filtro Butterworth', 'Frequência (rad/sample)', 'Amplitude', color='green')


#__________________________ Questão 13 _____________________________________

# Calcula a resposta em fase da resposta em frequência do filtro Butterworth 
phase_hFreqResponse = np.angle(hFreqResponse)

# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, len(phase_hFreqResponse))

plot_graph(x, phase_hFreqResponse, 'Resposta em Fase do filtro Butterworth', 'Frequência', 'Amplitude', color='green')

#__________________________ Questão 14 _____________________________________

# Calcula o atraso de grupo do filtro Butterworth 
frequencies, groupDelay = signal.group_delay((b, a))

# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, len(groupDelay))

plot_graph(x, groupDelay, 'Atraso de Grupo do filtro Butterworth', 'Frequência', 'Atraso de Grupo (samples)', color='green')