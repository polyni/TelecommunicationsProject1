import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from pylab import *

def sample_time(f, fs, Tnum):
    # Διάνυσμα χρόνου μέχρι <Tnum> περιόδους με δείγμα ανά 1/fs (sec)
    return np.arange(0, Tnum * 1/f, 1/fs)
def signal_sinim(A ,f ,t ):
    return A*np.sin(2*np.pi*f*t)
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