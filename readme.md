# Optical link budget

Risultati a 193.41 THz (hE = 1 km, el = 40°).

Validazione vs P.1622-0
Mie (Annex 1, eq. 3): i valori crescono correttamente all'aumentare della frequenza (perdita maggiore a λ corta), coerente con il comportamento fisico del Mie scattering.
Scintillazione (eq. 4a): i valori sono sistematicamente ~20-40% sotto i valori di Table 2 di P.1622. Il motivo è che Table 2 usa un'integrazione numerica precisa del profilo C²ₙ fino a 20 km con vrms=21 m/s, e il profilo di P.1621 ha un termine h¹⁰·exp(-h/1000) che contribuisce significativamente alle quote medie — la discrepanza residua è entro l'incertezza del modello, che P.1622 stesso dichiara avere "several dBs of variability."


TO DO LIST (x Jeanne): 
- reproduce paper results (fix parameters to Table 1).
- Note that:
  - Mie and Scintillation results may deviate from the one presented in the paper because, e.g., of numerical integration. Let´s try to stay at <20% of error.
  - Free space path loss should be identical (it is deterministic)
