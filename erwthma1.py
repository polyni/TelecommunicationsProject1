import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from pylab import *
from sympy.combinatorics.graycode import GrayCode

#erwthma 1a
fm  = 9000
fs1 = fm * 25
fs2 = fm * 60
fs3 = fm * 5
A = 4  #Amplitude

def sample_time(f, fs, n):
    # Διάνυσμα χρόνου μέχρι n περιόδους με δείγμα ανά 1/fs (sec)
    return np.arange(0, n * 1/f, 1/fs)
def signal_sq_triangle(A ,f ,t ):
    #συναρτηση που παράγει το τετράγωνο της τριγωνικής περιοδικής παλμοσειράς, με περίοδο f
    return A * pow(ss.sawtooth(2 * np.pi * f * t),2)
def signal_sinim(A ,f ,t ):
    # συνάρτηση που παράγει ημίτονο περιόδου f
    return A*np.sin(2*np.pi*f*t)

grid(True)
t = np.linspace(0, 4*1/fm, 10000)
sq_triangle = signal_sq_triangle(A,fm,t)
plt.plot(t,sq_triangle)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("sq_triangle(t)")
plt.show()

grid(True)
time1 = sample_time(fm,fs1,4) #το time1 περιέχει τις χρονικές στιγμές στις οποίες θα πρέπει να υπάρχουν δείγματα
y1 = signal_sq_triangle(A,fm,time1)
plt.plot(time1,y1,'.')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(i) Sampling with fs1=25fm")
plt.show()

grid(True)
time2 = sample_time(fm,fs2,4)
y2 = signal_sq_triangle(A,fm,time2)
plt.plot(time2, y2, '.r')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(ii) Sampling with fs2=60fm")
plt.show()

grid(True)
plt.plot(time1, y1, '.', label='Sampling with fs1')
plt.plot(time2, y2, '.r', label='Sampling with fs2')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(iii) Shared Diagram")
plt.legend(loc=1)
plt.show()

#erwthma 1b

time3 = sample_time(fm,fs3,4)
y3 = signal_sq_triangle(4,fm,time3)
plt.plot(time3, y3, '.g')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Sampling with 5fm")
plt.show()

#erwthma 1ci

grid(True)
y= signal_sinim(1,fm,t)
plt.plot(t,y)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("z(t)")
plt.show()

grid(True)
time1 = sample_time(fm,fs1,4)
# Κατασκευή διαγράμματος σήματος-χρόνου
y1=  signal_sinim(1,fm,time1)
plt.plot(time1,y1,'.')
# Τίτλος-Υπόμνημα-Λεζάντες
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(i) Sampling with fs1=25fm")
plt.show()

grid(True)
time2 = sample_time(fm,fs2,4)
y2 = signal_sinim(1,fm,time2)
plt.plot(time2, y2, '.r')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(ii) Sampling with fs2=60fm")
plt.show()

grid(True)
plt.plot(time1, y1, '.', label='Sampling with fs1')
plt.plot(time2, y2, '.r', label='Sampling with fs2')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(iii) Shared Diagram")
plt.legend(loc=1)
plt.show()

grid(True)
time3 = sample_time(fm,fs3,4)
y3 = signal_sinim(1,fm,time3)
plt.plot(time3, y3, '.g')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Sampling with 5fm")
plt.show()

#erwthma 1cii

fm2 = 10000
fd = 1000  #συχνότητα του q(t)
fs1 = fd * 25
fs2 = fd * 60
fs3 = fd * 5

grid(True)
# z = y + sin(2πt*fm2)
t = np.linspace(0, 0.001, 10000)
q = np.add(signal_sinim(1,fm,t), signal_sinim(1,fm2,t))
plt.plot(t, q)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("q(t)")
plt.show()


grid(True)
tz1 = sample_time(fd,fs1,1)
# z = y + sin(2πt*fm2)
z1  = np.add(signal_sinim(1,fm,tz1), signal_sinim(1,fm2,tz1))
plt.plot(tz1, z1, '.')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(i) Sampling with fs1")
plt.show()

grid(True)
tz2 = sample_time(fd,fs2,1)
z2  = np.add(signal_sinim(1,fm,tz2), signal_sinim(1,fm2,tz2))
plt.plot(tz2, z2, '.r')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(ii) Sampling with fs2")
plt.show()

grid(True)
plt.plot(tz2, z2, '.r', label='Sampling with fs2')
plt.plot(tz1, z1, '.',label='Sampling with fs1')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(iii) Shared Diagram")
plt.legend(loc=1)
plt.show()

grid(True)
tz3 = sample_time(fd,fs3,1)
z3  = np.add(signal_sinim(1,fm,tz3), signal_sinim(1,fm2,tz3))
plt.plot(tz3, z3, 'o')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Sampling with 5*fd")
plt.show()

# erwthma 2
# a meros

bits = 5
levels = 2 ** bits-1
delta = 4.0 / levels # απόσταση μεταξύ των levels
fm = 9000
fs = fm * 45
A = 4
time1 = sample_time(fm, fs, 4)
y = signal_sq_triangle(A, fm, time1)
y_quantized = np.zeros(len(y))
h = np.zeros(len(y))
for i in range(0, len(y)):
    h[i] = 2 * (y[i] // delta) + 1
    y_quantized[i] = (delta / 2) * h[i]  # mid riser, y_quantized = delta*([y/delta]+0.5)
y_axis = np.linspace(-4 + delta / 2, 4 - delta / 2, 31)
G_C = GrayCode(5)
grid(True)
plt.yticks(y_axis, G_C.generate_gray())

plt.stem(time1, y_quantized)
plt.xlabel("Time (sec)")
plt.ylabel("Quantizing Levels ( Natural Binary Coding 4bit )")
plt.title("Quantized Signal")
plt.show()

#b meros
def SNR(y, y_quantized, length):
    # η συνάρτηση υπολογίζει το Quantum error
    quantization_error = y_quantized - y
    average_error = 0
    for i in range(length):
        average_error += quantization_error[i]/length
    error_power = 0
    for i in range(length):
        error_power += ((quantization_error[i]-average_error)**2) / length
    print("Standard Deviation error for", length, "samples:", np.sqrt(error_power))
    signal_power = 0
    for i in range(length):
        signal_power += ((y[i])**2)/length
    SNR_exp = signal_power / error_power
    SNR_exp_dB = 10 * np.log10(SNR_exp)
    print("SNR for", length, "samples:", SNR_exp, "=", SNR_exp_dB, "dB\n")


SNR(y, y_quantized, 10)  # 10 samples
SNR(y, y_quantized, 20)  # 20 samples

#Theoritical SNR
var_error_theory = delta**2 / 12
signal_power_theory = sum(y**2)/len(y)
SNR_theory = (signal_power_theory/var_error_theory)
SNR_theory_dB = 10*np.log10(SNR_theory)
print ("Theoretical SNR: ", SNR_theory, "=", SNR_theory_dB, "dB" )
#erwthma c

G_C = G_C.generate_gray()
gray_list = list(G_C)
levels = np.rint(y_quantized/delta) #υπολογίζουμε το level στο οποίο βρίσκεται το i-οστο κβάντο
levels = [int(x) for x in levels]
gray_function = [0]*(np.size(levels))
for i in range(np.size(levels)):
    gray_function[i] = int(gray_list[levels[i]-1]) # η συνάρτηση gray_function περιέχει το level στο οποίο βρίσκεται το
       #i-οστο κβάντο σε κώδικα gray
sign = 1
bit_stream = [0]*(10*np.size(gray_function))
for i in range(np.size(gray_function)):
    for j in range(5):
        bit_stream[10*i+2*j] = (gray_function[i]//10**(4-j))%2*sign*9 #Η bit_stream έχει τα bits που προκύπτουν από την
          #gray_με κωδικοποίση γραμμής BIPOLAR RZ
        if ((gray_function[i]//10**(4-j))%2 == 1):
            sign = sign*(-1)
        bit_stream[10*i+2*j+1] = 0
t = np.linspace(0, 0.0005*50, 50*100) #θα παρουσιάσουμε τα πρώτα 25 bits
sig_transmitted = [0]*np.size(t)
for i in range(np.size(t)):
    sig_transmitted[i] = bit_stream[i//100]
plt.figure(figsize=(20, 15))
plt.plot(t, sig_transmitted)
plt.grid()
plt.xlabel ("Time(s)")
plt.ylabel ("Voltage(V)")
plt.title("Bit stream for the first 25 bits")
plt.show()

#erwthma 3a
fs = 130*9000
ttel = 4*(1/35)
t = np.linspace(0, ttel, 1000)
m = signal_sinim(1,35, t)           # σήμα πληροφορίας
c = signal_sinim(1,fs, t)             # φέρον σήμα σε διαμόρφωση
s = (1.0 + 0.5*m)*c           # διαμορφωμένο σήμα


grid(True)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Modulated Signal")
plt.plot(t,s)
plt.show()

#erwthma 3b

s2 = s*c
b_cutoff = 0.1

n = 100 # Order
# H(ω) = a(ω)/b(ω)
a = 1
b = ss.firwin(n, cutoff=b_cutoff)
after_filter = ss.lfilter(b, a, s2)
#Καθυστερήσεις
delay = 80
v_out = after_filter[delay:]
t_new = t[0:(len(t)-delay)]
# Κάθε πολλαπλασιασμός cos υποδιπλασιάζει το πλάτος => *4
# Aφαιρούμε και τη dc συνιστώσα => τελική έξοδος
y_out = 4*(v_out - average(v_out))


plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("demodulated signal m(t)")
plt.plot(t_new, y_out, linewidth = 4, label = 'Demodulated Signal')
plt.grid(True)
plt.show()