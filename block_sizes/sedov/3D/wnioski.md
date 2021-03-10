Wykres w gnuplocie:

    set log x
    plot [7:400] (1 + 24/x)*0.000006, "MHD" u 2:($3/$1**3/2.3) w lp, "HD" u 2:($3/$1**3) w lp  


dla symulacji 3D sedova z solverem Riemanna (2D_RIEMANN wygląda podobnie)
pokazuje, że:

* Jest jakaś wybitna nieoptymalnośc z rozmiarem bloku 120^3.  Wyczerpanie
  pojemności L3 ?
* Jest spadek wydajności przy przejściu z 16^3 na 20^3. Wyczerpanie L1 + L2?
* Ten spadek jest bardziej widoczny dla czystego HD a mniej dla MHD

Zapewne dałoby się (1 + 24/x) zredukować.  Być może nawet do (1 + 15/x) (25%
dla bloków 16^3), gdyby omijać niepotrzebne obliczenie w guardcellkach. 
Dobrze że nie jest to (x+8)^3/x^3 (memory overhead).
