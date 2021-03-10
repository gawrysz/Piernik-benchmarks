Wykres w gnuplocie:

    set log x
    plot [7:] (1 + 16/x)*0.0000047, "MHD" u 2:($3/$1**2/2.3) w lp, "HD" u 2:($3/$1**2) w lp

dla symulacji 2D sedova z solverem Riemanna (2D_RIEMANN wygląda podobnie)
pokazuje, że:

* Jest jakaś wybitna nieoptymalnośc z rozmiarem bloku 504^2.  Wyczerpanie
  pojemności L3 ?
* Jest jakaś wybitna nieoptymalnośc z rozmiarem bloku 20^2. Znów cache?
* Z jakichś powodów wydajność dla dekompozycji 4-blokowej jest lepsza niż
  dla 1-blokowej, o ile rozmiar jest powyżej ~800^2
