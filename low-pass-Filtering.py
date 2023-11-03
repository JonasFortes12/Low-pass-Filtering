import numpy as np
import math as mt
import matplotlib.pyplot as plt
from scipy import signal

# Função para plotagem de gráficos
def plot_graph(x, y,title, xlabel, ylabel, y2=None, label=None,label2=None, color='blue', color2='red', mode='plot'):
    plt.figure(figsize=(8, 4))
    
    if(mode == 'plot'):
        plt.plot(x, y, label=label, color=color)
        if(y2 is not None):
            plt.plot(x, y2, label=label2, color=color2)
    elif(mode == 'stem'):
        plt.stem(x, y, label=label)
        if(y2 is not None):
            plt.stem(x, y2, label=label2)
    
    # Adicionar um título ao gráfico
    plt.title(title)
    # Adicionar rótulos aos eixos
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    if(label is not None):
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
alpha = 15  # Atraso
# Tamanho da resposta ao impulso
M = (2 * alpha) + 1  # M = 30 

# Amostras de 0 até M (31 amostras)
n = np.arange(M)

# Resposta ao impulso do filtro passa-baixa FIR
h_ideal = np.sin(cutOffFreq * (n-alpha)) / (np.pi * (n-alpha))
h_ideal[alpha] = 1 # Tratar a divisão por zero (definição do sinc)


plot_graph(n, h_ideal, 'Resposta ao Impulso do Filtro Passa-Baixa FIR h[n]', 'Tempo', 'Amplitude', mode='stem')


#__________________________ Questão 04 _____________________________________

# Calcula a resposta em frequência do sistema (FT da reposta ao impulso h[n])
h_idealFT = np.fft.fftshift(np.fft.fft(h_ideal))

# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, M)

plot_graph(x, np.abs(h_idealFT), 'Módulo da Resposta em Frequência do Sistema', 'Frequência', 'Amplitude', color='black')

#__________________________ Questão 05 _____________________________________

# Calcula a resposta em fase da resposta em frequência do sistema
phase_h_idealFT = np.unwrap(np.angle(h_idealFT))

plot_graph(x, phase_h_idealFT, 'Resposta em Fase do Sistema', 'Frequência', 'Amplitude', color='green')


#__________________________ Questão 06 _____________________________________

# Calcula o atraso de grupo do sistema 
frequencies, groupDelay = signal.group_delay((h_ideal, 1))

# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, len(groupDelay))

plot_graph(x, groupDelay, 'Atraso de Grupo do Sistema', 'Frequência', 'Atraso de Grupo (samples)')


#__________________________ Questão 07 _____________________________________

# Realiza a convolução discreta para filtrar o sinal x[n] com a resposta ao impulso h[n]
Yn = np.convolve(Xn, h_ideal) 
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

N = len(Yn)  # Número de amostras
n = np.arange(N)  # Vetor de amostras de 0 a N-1

# Frequências desejadas em radianos por amostra
frequencies = [0.1 * mt.pi, 0.6 * mt.pi]

# Gera o sinal somando as componentes de frequência
Gn = np.sum([np.cos(omega * n) for omega in frequencies], axis=0)

alpha = 15

# Aplica um atraso em G: G[n-alpha]
Gn_offset = np.roll(Gn, alpha)

plot_graph(n, Gn_offset, 'Sinais G[n] e Y[n-a]', 'Tempo', 'Amplitude', Yn, 'Sinal g[n-alpha]', 'Sinal y[n]')

#__________________________ Questão 10 _____________________________________
"""
    O filtro FIR (Finite Impulse Response) tem como vantagem uma resposta em frequência precisa e nítida,
pois possue uma resposta em frequência caracterizada por uma magnitude retangular (idealmente).
Isso significa que a magnitude da resposta é constante na faixa de passagem e cai abruptamente
para zero na faixa de rejeição, como é notado no gráfico da questão 04. Outra vantagem é que sua resposta 
em fase é previsível (linear), como é percebido nn gráfico da questão 05. Também, o atraso de grupo é 
constante, como mostra o gráfico da questão 06. Essas vantagens possibilitam que esse filtro seja facilmente
projetado, apresentando ótima estabilidade e menos suscetível a problemas. 

A desvantagem é que requer um número maior de coeficientes para alcançar respostas em frequência complexas. 
Assim, pode exigir mais recursos computacionais para implementação em tempo real
"""

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

plot_graph(x, np.abs(hFreqResponse), 'Módulo da Resposta em Frequência do Filtro Butterworth', 'Frequência', 'Amplitude', color='green')


#__________________________ Questão 13 _____________________________________

# Calcula a resposta em fase da resposta em frequência do filtro Butterworth 
phase_hFreqResponse = np.unwrap(np.angle(hFreqResponse))

# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, len(phase_hFreqResponse))

plot_graph(x, phase_hFreqResponse, 'Resposta em Fase do filtro Butterworth', 'Frequência', 'Amplitude', color='green')

#__________________________ Questão 14 _____________________________________

# Calcula o atraso de grupo do filtro Butterworth 
frequencies, groupDelay = signal.group_delay((b, a), x, whole=True)

# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, len(groupDelay))

plot_graph(x, groupDelay, 'Atraso de Grupo do filtro Butterworth', 'Frequência', 'Atraso de Grupo (samples)', color='green')

#__________________________ Questão 15 _____________________________________

# Realiza a convolução discreta para filtrar o sinal x[n] com o filtro Butterworth 
Yn_Butterworth = signal.lfilter(b, a, Xn)
n = np.arange(len(Yn_Butterworth))

plot_graph(n, Yn_Butterworth,  'Sinal de Saída y[n] ( x[n] * Butterworth Filter )', 'Tempo', 'Amplitude')


#__________________________ Questão 16 _____________________________________

# Calcula o módulo da transformada de Fourier do sinal de saída y[n] pelo Butterworth
Yn_ButterworthFT = np.abs(np.fft.fftshift(np.fft.fft(Yn_Butterworth)))

# Definir valores do eixo X
x =  np.linspace(-np.pi, np.pi, len(Yn_ButterworthFT))

plot_graph(x, Yn_ButterworthFT, 'Módulo da FT do sinal Y[n] pelo Butterworth', 'Frequência', 'Amplitude', color='black')

#__________________________ Questão 17 _____________________________________

# Gerando o sinal g[n] com 100 amostras
N = len(Yn_Butterworth)  # Número de amostras
n = np.arange(N)  # Vetor de amostras de 0 a N-1

# Frequências desejadas em radianos por amostra
frequencies = [0.1 * mt.pi, 0.6 * mt.pi]

# Gera o sinal somando as componentes de frequência
Gn = np.sum([np.cos(omega * n) for omega in frequencies], axis=0)

# Aplica um atraso em y: y[n-nd]
alpha = -2 # atraso nd
Yn_Butterworth_offset = np.roll(Yn_Butterworth, alpha);

plot_graph(n, Gn, 'Sinais G[n] e Y[n-nd] (Butterworth)', 'Tempo', 'Amplitude', Yn_Butterworth_offset, 'Sinal g[n]', 'Sinal y[n-a]')

#__________________________ Questão 18 _____________________________________
'''
    O filtro IIR (Infinite Impulse Response), Butterworth, apresenta a vantagem de uma resposta em frequência 
suave,ou seja, sua resposta em frequência tende a ser menos abrupta na transição entre a faixa de passagem 
e a de rejeição, como é observado no gráfico da questão 12.
    Tem a desvantagem de ter uma resposta em fase não linear devido à presença de polos. Isso pode levar a 
distorçõesde fase, especialmente perto das frequências de corte. A resposta em fase do filtro não é linear
em toda a faixa de frequência, como é observado no gráfico da questão 13.
    Outra desvantagem desse filtro é ter um atraso de grupo variável, o que significa que diferentes componentes
de frequência do sinal podem ser atrasados de maneira diferente. Isso pode levar a distorções na forma de onda
do sinal, pois diferentes partes do sinal são atrasadas de maneira desigual. 
Como é observado no gráfico da questão 14.
'''