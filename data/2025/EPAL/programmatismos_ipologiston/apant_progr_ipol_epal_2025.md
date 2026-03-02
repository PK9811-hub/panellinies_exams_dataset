A1.α    Σωστό
A1.β    Σωστό
A1.γ    Λάθος
A1.δ    Λάθος
A1.ε    Σωστό
A2  ["1δ", "2γ", "3β", "4α", "5στ"]
B1.α    "```python\ndef max_poso(self):\n    max = self.poliseis[0]\n    for x in self.poliseis:\n        if x > max:\n            max = x\n    return max"    
B1.β    "```python\npolitis1 = Politis("Ιωάννα Κωνσταντίνου",[10000,15000,5000,20000])"
B1.γ    "```python\nprint(politis1.max_poso())"
B2  "1. ''\n2. lexi\n3. arxika\n4. 0\n5. arxika"
B3  "1. ΠΡΟΤΥΠΟ\n2. ΠΡΟΤΥΠΟ ΕΠΑΛ\n3. ΤΥΠΟ ΕΠΑ\n4. ΕΠΑΛ\n5. ΠΕ"
Γ1  "```python\nSXOL = []\nTYPOS = []\nMATH = []\n\nfor i in range(5):\n    for j in range(4):\n        SXOL.append(str(input('Δώσε όνομα σχολείου')))\n        TYPOS.append(str(input('Δώσε τύπο σχολείου')))\n        plithos = int(input('Δώσε πλήθος μαθητών'))\n        while plithos < 20 or plithos > 50:\n            plithos = int(input('Λάθος τιμή. Ξαναδώσε'))\n        MATH.append(plithos)"
Γ2  "```python\nfor i in range(20):\n    COST = Ypologismos(MATH[i])\n    print('Το κόστος των εισιτηρίων του σχολείου είναι:',COST,'€')"
Γ3  "```python\ndef YPOLOGISMOS(x):\n    return 4*x"
Γ4  "```python\nSum = 0\nplithos = 0\nPI_PEPAL \nfor i in range(20):\n    Sum += COST\n    plithos += MATH[i]\n    if TYPOS[i] == 'ΠΕΠΑΛ':\n        PI_PEPAL += MATH[i]\nprint('Συνολικά έσοδα:',Sum,'€')\npososto = PI_PEPAL/float(plithos)\nprint('Ποσοστό μαθητών ΠΕΠΑΛ:',pososto,'%')"
Δ1  "```python\n\nANS = ['a', 'd', 'b', 'b', 'a', 'c', 'd', 'a', 'b', 'c']\nKOD = []\nSB = []\nfor i in range(300):\n    KOD.append(str(input('Δώσε κωδικό υποψηφίου')))\n    sum = 0\n    for j in range(10):\n        apant = str(input('Δώσε απάντηση υποψηφίου'))\n        if apant == ANS[j]:\n            sum += 5\n    SB.append(sum)"
Δ2  "```python\nfor i in range(300):\n    if SB[i] > 30:\n        print(KOD[i])"
Δ3.α    "```python\nfor i in range(0,299,1):\n    for j in range(299,i,-1):\n        if SB[j] > SB[j-1]:\n            SB[j], SB[j-1] = SB[j-1], SB[j]\n            KOD[j], KOD[j-1] = KOD[j-1], KOD[j]"
Δ3.β    "```python\nf = open('lang.txt','w')\nfor i in range(300):\n    text = KOD[i] + " " + str(SB[i])\n    f.write(text + "\n")\nf.close()"