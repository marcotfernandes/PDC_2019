function speedUp = Amdahl_Law(Tbefore,Taffected,Tunnaffected,N)
    speedUp = Tbefore/(Taffected/N + Tunnaffected);
end

