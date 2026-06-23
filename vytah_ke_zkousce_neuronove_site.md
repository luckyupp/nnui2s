# Výtah ke zkoušce: umělé neuronové sítě

Studijní výtah je postavený podle skript `Úvod do umělých neuronových sítí (1).pdf` a podle témat ke zkoušce. Cílem není jen vyjmenovat pojmy, ale pochopit, proč jednotlivé sítě vznikly, k čemu se hodí, jak fungují a jaké mají limity. U důležitých pojmů je uveden anglický ekvivalent, protože terminologie se v praxi i literatuře často používá anglicky.

Konvoluční neuronové sítě jsou doplněny samostatně, protože ve skriptech jsou pouze zmíněny mezi dalšími topologiemi, ale nejsou samostatně vyloženy.

## 0. Celkový obraz problematiky

Umělá neuronová síť (artificial neural network, ANN) je výpočetní model složený z jednoduchých jednotek, umělých neuronů (artificial neurons), které jsou propojené váženými spoji. Síť přijímá vstupní data, uvnitř je transformuje a vytváří výstup. To nejdůležitější není samotný výpočet jednoho neuronu, ale schopnost nastavit parametry sítě z dat. Tomu říkáme učení (learning, training).

Klasické programování typicky postupuje tak, že člověk napíše pravidla. Například: pokud je intenzita světla vysoká a hmotnost lupínku dostatečná, výrobek je vyhovující. Neuronová síť postupuje jinak. Dostane mnoho příkladů vstupů a správných výstupů a sama si nastaví váhy tak, aby podobné případy řešila podobně. Není to tedy systém založený hlavně na explicitních pravidlech, ale na aproximaci vztahu mezi vstupy a výstupy.

Obecná představa:

- vstupní veličiny (inputs) popisují objekt nebo stav systému;
- váhy (weights) určují důležitost jednotlivých vstupů;
- neuron spočítá vážený součet a převede ho aktivační funkcí;
- vrstvy neuronů skládají jednoduché transformace do složitějších;
- učení mění váhy a prahy tak, aby síť dávala požadované odpovědi;
- kvalita sítě se ověřuje na datech, která síť při učení nepoužila.

Neuronová síť tedy realizuje transformační funkci. Matematicky lze říct, že se snaží naučit zobrazení:

```text
y = F(x)
```

kde `x` je vstupní vektor (input vector), `y` je výstupní vektor (output vector) a `F` je funkce realizovaná topologií sítě a jejími naučenými vahami.

Pro pochopení celé látky je užitečné držet si v hlavě následující rozdělení:

- jednoduchý perceptron: nejjednodušší klasifikátor, lineární rozhodování;
- Hopfieldova síť: rekurentní autoasociativní paměť;
- Kohonenova mapa: samoorganizující shlukování bez učitele;
- vícevrstvá dopředná síť: obecný aproximátor, klasifikátor, prediktor;
- konvoluční síť: specializovaná vícevrstvá síť pro obrazy a prostorová data.

## 1. Umělé neuronové sítě, členění, definice pojmů

### 1.1 Biologická inspirace a rozdíl proti biologii

Umělé neuronové sítě jsou inspirovány biologickým nervovým systémem, ale nejsou jeho věrnou kopií. Biologický neuron přijímá signály dendrity, zpracovává je v těle neuronu a posílá výstup axonem přes synapse. Umělý neuron tuto strukturu zjednodušuje na matematický model:

- dendrity odpovídají vstupům;
- synapse odpovídají váhám spojení;
- tělo neuronu odpovídá agregaci a aktivaci;
- axon odpovídá výstupu.

Toto přirovnání je důležité pro intuici, ale u zkoušky je dobré říct, že umělý neuron je především matematický výpočetní prvek. Většina běžných neuronových sítí nepředstírá biologickou přesnost. Cílem je výpočetní užitečnost: klasifikace, aproximace, predikce, modelování, řízení nebo rozpoznávání vzorů.

### 1.2 Formální neuron

Základní model neuronu se často nazývá formální neuron (formal neuron) nebo McCulloch-Pittsův neuron (McCulloch-Pitts neuron). Má konečný počet vstupů a jeden výstup.

Vstupy:

```text
x1, x2, ..., xR
```

Váhy:

```text
w1, w2, ..., wR
```

V základním modelu se nejprve spočítá vstupní potenciál:

```text
ya = Σ wi xi
```

Při zahrnutí prahu nebo biasu:

```text
ya = Σ wi xi, i = 0 až R
x0 = 1
```

Výstup neuronu:

```text
y = φ(ya)
```

kde `φ` je aktivační funkce (activation function).

### 1.3 Vstupy

Vstup (input) je hodnota přiváděná do neuronu. V neuronových sítích bývají vstupy nejčastěji čísla. I nečíselné informace, jako třída, text nebo obrázek, se musí převést na číselnou reprezentaci.

Vstupy mohou být:

- kvalitativní (qualitative): například 0/1 pro ne/ano;
- kvantitativní (quantitative): reálné hodnoty, například teplota, tlak, jas pixelu;
- výstupy jiných neuronů;
- zpožděné hodnoty signálu u dynamických modelů.

Vstupní vektor (input vector) je uspořádaná sada vstupů:

```text
x = [x1, x2, ..., xR]^T
```

U inženýrských úloh je velmi důležité, jak vstupy zvolíme. Síť se může naučit jen vztahy obsažené v datech. Pokud do vstupů nedáme relevantní informaci, síť ji nemůže vymyslet. Například při modelování dynamického systému nestačí dát pouze aktuální vstup, pokud výstup závisí i na minulých stavech.

### 1.4 Váhy spojení

Váha spojení (connection weight, synaptic weight) je parametr, který říká, jak silně daný vstup působí na neuron.

Interpretace:

- kladná váha zvyšuje vstupní potenciál;
- záporná váha ho snižuje;
- váha blízko nule znamená, že vstup má malý vliv;
- velikost váhy souvisí s důležitostí daného spojení.

Učení neuronové sítě je ve většině případů právě adaptace vah (weight adaptation). Síť tedy nehledá pravidla v symbolické podobě, ale hodnoty parametrů.

### 1.5 Práh a bias

Práh neuronu (threshold) je hodnota, kterou musí vstupní signál překonat, aby byl neuron aktivní. V moderní terminologii se často používá pojem bias. Bias je posun aktivační hranice neuronu.

Proč je bias důležitý:

- umožňuje posunout rozhodovací hranici;
- bez biasu by hranice často musela procházet počátkem souřadnic;
- zvyšuje flexibilitu modelu.

Prakticky se bias zapisuje jako váha `w0` připojená k fiktivnímu vstupu `x0 = 1`.

```text
ya = w0 x0 + w1 x1 + ... + wR xR
```

### 1.6 Agregační funkce

Agregační funkce (aggregation function) slučuje vstupní signály do jedné skalární hodnoty. V základním modelu je to vážený součet.

```text
ya = Σ wi xi
```

Tento vztah je jednoduchý, ale silný. Samotný vážený součet je lineární operace. Síla neuronových sítí vzniká až kombinací mnoha takových operací s nelineárními aktivačními funkcemi.

### 1.7 Aktivační funkce

Aktivační funkce (activation function, transfer function) převádí vstupní potenciál na výstup neuronu.

```text
y = φ(ya)
```

Její význam:

- zavádí nelinearitu;
- omezuje rozsah výstupu;
- umožňuje neuronům rozhodovat nebo hladce aproximovat;
- ovlivňuje učitelnost sítě.

Typické aktivační funkce:

Skoková funkce (step function):

```text
y = 1, pokud ya >= 0
y = 0 nebo -1, pokud ya < 0
```

Používá se u perceptronu a binárních rozhodovacích neuronů. Nevýhoda je nediferencovatelnost, takže se nehodí pro klasický backpropagation.

Lineární funkce (linear function):

```text
y = ya
```

Hodí se často ve výstupní vrstvě regresních sítí, kde chceme libovolnou reálnou hodnotu.

Sigmoidální funkce (sigmoid function):

```text
y = 1 / (1 + e^(-ya))
```

Výstup je v intervalu 0 až 1. Používala se často v neuronových sítích, ale u hlubokých sítí může trpět saturací a malými gradienty.

Hyperbolický tangens (hyperbolic tangent, tanh):

```text
y = tanh(ya)
```

Výstup je v intervalu -1 až 1. Ve skriptech je často doporučován pro skryté vrstvy, protože je lichý a data normalizovaná na `<-1; 1>` s ním dobře pracují.

ReLU (rectified linear unit):

```text
y = max(0, ya)
```

Ve skriptech není hlavním tématem, ale u moderních CNN a hlubokých sítí je zásadní. Je výpočetně jednoduchá a pomáhá trénovat hlubší sítě.

Důležité pochopení: pokud by všechny aktivační funkce byly lineární, vícevrstvá síť by se dala matematicky zredukovat na jednu lineární vrstvu. Nelinearita je důvod, proč síť dokáže modelovat složité závislosti.

### 1.8 Topologie sítě

Topologie nebo architektura sítě (network topology, architecture) popisuje, jak jsou neurony uspořádány a propojeny.

Základní vrstvy:

- vstupní vrstva (input layer): přivádí signál do sítě;
- skrytá vrstva (hidden layer): vnitřní výpočetní vrstva;
- výstupní vrstva (output layer): vytváří finální odpověď.

Vstupní vrstva se často nepovažuje za aktivní vrstvu neuronů, protože pouze předává hodnoty dál.

Podle směru šíření signálu:

- dopředné sítě (feedforward networks): signál jde pouze od vstupu k výstupu;
- rekurentní sítě (recurrent networks): existuje zpětná vazba a stav v čase;
- laterálně propojené sítě (lateral connections): propojení uvnitř vrstvy, typicky u kompetitivních struktur.

Topologie určuje výpočetní schopnosti sítě. Jednoduchý perceptron umí jen lineární hranici. Vícevrstvá síť umí nelineární hranice. Hopfieldova síť umí pracovat jako paměť. Kohonenova mapa umí organizovat vstupní prostor do shluků.

### 1.9 Učení neuronových sítí

Učení (learning, training) je proces změny nastavitelných parametrů sítě tak, aby síť dosáhla požadovaného chování. Nejčastěji se mění váhy a biasy.

Učení s učitelem (supervised learning):

- máme vstupy a správné výstupy;
- síť vytvoří výstup;
- výstup se porovná s cílem;
- podle chyby se upraví váhy.

Typické úlohy: klasifikace, regrese, predikce.

Učení bez učitele (unsupervised learning):

- máme pouze vstupy;
- síť hledá podobnosti, strukturu nebo shluky;
- neexistuje cílový výstup `t`.

Typické úlohy: shlukování, redukce dimenze, vizualizace dat.

Hebbovo učení (Hebbian learning):

- propojení aktivních neuronů se posiluje;
- jde o historicky důležitý princip;
- ve skriptech je spojeno hlavně s jednoduchým perceptronem a Hopfieldovou sítí.

Chybové učení (error learning):

- používá chybu mezi cílem a výstupem;
- typické pro perceptron a vícevrstvé dopředné sítě.

### 1.10 Datové množiny a generalizace

Trénovací množina (training set) slouží k nastavování vah. Validační množina (validation set) slouží ke kontrole během vývoje, například k výběru topologie nebo zastavení učení. Testovací množina (test set) slouží až na finální ověření.

Generalizace (generalization) je schopnost sítě správně reagovat na data, která při tréninku neviděla. To je zásadní. Síť, která si pouze zapamatuje trénovací data, může být v praxi špatná.

Přeučení (overfitting) znamená, že síť má velmi malou chybu na trénovacích datech, ale špatně funguje na nových datech. Typicky nastává, když je síť příliš složitá vzhledem k množství dat.

Podučení (underfitting) znamená, že síť je příliš jednoduchá nebo špatně naučená a nedokáže zachytit ani základní vztah v trénovacích datech.

Normalizace (normalization) je převod dat do vhodného rozsahu, například `<-1; 1>` nebo `<0; 1>`. Je důležitá, protože výrazně rozdílná měřítka vstupů mohou zhoršit učení.

## 2. Jednoduchý perceptron

### 2.1 Co je perceptron

Jednoduchý perceptron (simple perceptron, single-layer perceptron) je nejjednodušší neuronová síť pro rozhodování mezi dvěma třídami. Je tvořen jedním formálním neuronem s lineárně váženou agregační funkcí a skokovou aktivační funkcí.

Jeho výstup je typicky:

```text
y ∈ {0, 1}
```

nebo bipolárně:

```text
y ∈ {-1, 1}
```

Perceptron je důležitý historicky i didakticky, protože ukazuje základní princip učení vah podle chyby.

### 2.2 K čemu slouží

Perceptron řeší binární klasifikaci (binary classification). Má rozhodnout, zda vstup patří do třídy A nebo B.

Příklady:

- výrobek vyhověl/nevyhověl;
- pacient pozitivní/negativní;
- signál patří/nepatří do dané skupiny;
- jednoduché logické funkce AND nebo OR.

Perceptron vytvoří rozhodovací hranici (decision boundary). V dvourozměrném prostoru je to přímka, ve třech rozměrech rovina, ve vyšších dimenzích nadrovina.

### 2.3 Matematický model

Vstupní potenciál:

```text
ya = Σ wi xi
```

nebo:

```text
ya = w^T x
```

Výstup:

```text
y = φ(ya)
```

Pro bipolární skokovou funkci:

```text
y =  1, pokud ya >= 0
y = -1, pokud ya < 0
```

Rozhodovací hranice je dána rovnicí:

```text
w^T x = 0
```

Bias posouvá tuto hranici v prostoru. Bez biasu by hranice musela procházet počátkem, což je často zbytečné omezení.

### 2.4 Lineární separovatelnost

Lineární separovatelnost (linear separability) znamená, že dvě třídy lze oddělit jednou lineární hranicí.

Perceptron umí vyřešit pouze lineárně separovatelné úlohy. To je jeho hlavní omezení.

Funkce AND je lineárně separovatelná. Funkce OR je lineárně separovatelná. Funkce XOR není lineárně separovatelná, protože kladné a záporné příklady leží v prostoru tak, že je jedna přímka neoddělí.

Zkoušková formulace: jednoduchý perceptron není schopen realizovat XOR, protože jeho rozhodovací plocha je pouze lineární.

### 2.5 Učení perceptronu

Perceptron se učí s učitelem (supervised learning). Máme trénovací dvojice:

```text
(x, t)
```

kde `t` je požadovaný výstup (target).

Chyba:

```text
e = t - y
```

Úprava váhy:

```text
Δwi = α xi e
wi = wi + Δwi
```

kde `α` je koeficient rychlosti učení (learning rate).

Intuice:

- pokud perceptron odpoví správně, chyba je nulová a váhy se nezmění;
- pokud odpoví špatně, váhy se posunou směrem, který má příště zvýšit šanci správné odpovědi;
- vstupy s větší hodnotou způsobí větší změnu příslušné váhy.

Algoritmus:

1. Inicializuj váhy, často malými náhodnými čísly.
2. Zvol rychlost učení `α`.
3. Předlož vstupní vzor.
4. Spočítej výstup perceptronu.
5. Spočítej chybu.
6. Uprav všechny váhy.
7. Opakuj přes trénovací data po epochách.
8. Ukonči po dosažení chyby, počtu epoch nebo stabilizace.

### 2.6 Praktické poznámky

Rychlost učení ovlivňuje průběh trénování. Příliš malá hodnota vede k pomalému učení. Příliš velká hodnota může způsobit přeskakování dobrého řešení.

Pokud jsou data lineárně separovatelná, perceptronový algoritmus může konvergovat k řešení. Pokud separovatelná nejsou, dokonalé řešení neexistuje a perceptron bude dělat chyby.

V praktických úlohách se proto musí výsledek ověřit na testovacích datech. Není důležité jen to, že se váhy nějak nastavily, ale zda rozhodování funguje na nových případech.

### 2.7 Co umět říct u zkoušky

Jednoduchý perceptron je jeden neuron s váženým součtem a skokovou aktivační funkcí. Slouží k binární klasifikaci lineárně separovatelných dat. Učí se s učitelem úpravou vah podle chyby `e = t - y`. Jeho hlavním omezením je lineární rozhodovací hranice, takže neumí řešit nelineárně separovatelné problémy jako XOR.

## 3. Hopfieldova síť

### 3.1 Co je Hopfieldova síť

Hopfieldova síť (Hopfield network) je rekurentní neuronová síť (recurrent neural network) s jednou vrstvou neuronů. Každý neuron je propojen s každým jiným neuronem, ale ne sám se sebou.

Charakteristické vlastnosti:

- jednovrstvá síť;
- úplné propojení mezi neurony kromě vlastních smyček;
- zpětná vazba;
- stav sítě se vyvíjí v čase;
- často bipolární hodnoty `-1` a `1`;
- symetrická matice vah.

Oproti perceptronu zde nejde o jednorázové rozhodnutí z externího vstupu. Síť má vnitřní stav a opakovaně se přepočítává, dokud se neustálí.

### 3.2 K čemu slouží

Hopfieldova síť slouží hlavně jako autoasociativní paměť (autoassociative memory). Autoasociativní znamená, že vstup i výstup jsou stejného typu. Síť se naučí vzory a potom se z poškozené verze snaží vybavit původní vzor.

Příklad:

- uložíme několik binárních obrázků;
- vstupem je zašuměný obrázek;
- síť se iteracemi ustálí na uloženém obrázku, který je vstupu nejbližší.

Toto je podobné lidské schopnosti rozpoznat neúplný nebo poškozený tvar. Například i když je písmeno částečně zakryté, poznáme ho podle podobnosti s uloženou představou.

Hopfieldovy sítě se také používají jako modely pro optimalizační úlohy, ale ve skriptech je hlavní důraz na autoasociativní paměť.

### 3.3 Stav a váhy

Stav sítě je vektor výstupů neuronů:

```text
y = [y1, y2, ..., yR]^T
```

Typicky:

```text
yi ∈ {-1, 1}
```

Matice vah:

```text
W = [wij]
```

Platí:

```text
wii = 0
wij = wji
```

Nulová diagonála znamená, že neuron není spojen sám se sebou. Symetrie vah je důležitá pro stabilní chování sítě.

### 3.4 Vybavování vzoru

Vybavování (recall, retrieval) je proces, kdy síť z počátečního stavu dojde k nějakému stabilnímu stavu.

Pro první krok se použije vstupní vektor `x`. V dalších krocích se používá předchozí výstup sítě.

Vstupní potenciál neuronu je součet vlivů ostatních neuronů:

```text
ya,j(k) = Σ wij yi(k-1)
```

Výstup:

```text
yj(k) = φ(ya,j(k))
```

U bipolární skokové funkce je výstup `1` nebo `-1`.

Aktualizace může být:

- synchronní (synchronous update): všechny neurony najednou;
- asynchronní (asynchronous update): neurony po jednom.

Síť se zastaví, když:

- stav se už nemění;
- nebo osciluje mezi stavy;
- nebo je dosažen limit iterací.

### 3.5 Učení Hopfieldovy sítě

Učení Hopfieldovy sítě je jednorázové a vychází z Hebbova zákona. Vzory se ukládají do vah.

Pro `N` vzorů `t(p)`:

```text
wij = Σ ti(p) tj(p), pro i ≠ j
wij = 0, pro i = j
```

Intuice:

- pokud mají dvě pozice ve více vzorech stejnou hodnotu, jejich spojení se posiluje;
- pokud se často liší, spojení bude jiné;
- celá matice vah obsahuje informaci o uložených vzorech.

Na rozdíl od backpropagation zde neprobíhá dlouhé iterativní minimalizování chyby. Váhy se vypočítají přímo z ukládaných vzorů.

### 3.6 Stabilní stavy a energie

Hopfieldova síť se často interpretuje pomocí energetické funkce (energy function). Síť se při aktualizaci snaží přejít do stavu s nižší energií, podobně jako fyzikální systém směřuje do stabilního stavu.

Uložené vzory odpovídají atraktorům (attractors), tedy stabilním stavům. Pokud je počáteční vstup v oblasti přitažlivosti daného atraktoru, síť do něj konverguje.

Toto je dobrá intuice ke zkoušce: Hopfieldova síť není běžný dopředný klasifikátor. Je to dynamický systém, který hledá stabilní stav.

### 3.7 Kapacita a problémy

Kapacita Hopfieldovy sítě je omezená. Ve skriptech je uveden přibližný vztah:

```text
N < 0,138 R
```

kde `N` je počet uložených vzorů a `R` je počet neuronů.

Problémy:

- nízká kapacita;
- podobné vzory se mohou zaměňovat;
- síť může skončit ve fantomovém vzoru;
- síť může oscilovat;
- při příliš velkém šumu nemusí najít správný vzor.

Fantomový stav (spurious state) je stabilní stav, který nebyl mezi trénovacími vzory. Vzniká jako vedlejší efekt uložených vzorů.

### 3.8 Co umět říct u zkoušky

Hopfieldova síť je rekurentní jednovrstvá plně propojená síť se symetrickými vahami a nulovou diagonálou. Používá se jako autoasociativní paměť. Učí se jednorázově Hebbovým pravidlem a při vybavování iterativně konverguje ke stabilnímu stavu, který by měl odpovídat nejbližšímu uloženému vzoru. Její nevýhody jsou omezená kapacita, možnost fantomových stavů a oscilace.

## 4. Kohonenova samoorganizační mapa

### 4.1 Co je Kohonenova mapa

Kohonenova samoorganizační mapa (Kohonen self-organizing map, self-organizing map, SOM) je neuronová síť učená bez učitele. Patří mezi samoorganizující se mapy. Používá kompetitivní učení (competitive learning), při kterém neurony soutěží o to, který nejlépe reprezentuje aktuální vstup.

Na rozdíl od perceptronu nebo vícevrstvé sítě zde nemáme správný výstup. Síť se sama snaží uspořádat podle struktury vstupních dat.

### 4.2 K čemu slouží

SOM se používá hlavně pro:

- shlukovou analýzu (cluster analysis);
- rozpoznávání vzorů (pattern recognition);
- vizualizaci vícerozměrných dat;
- kompresi dat;
- hledání typických reprezentantů dat;
- předzpracování dat pro další metody.

Příklad: máme mnoho zákazníků popsaných věkem, příjmem, spotřebou a dalšími rysy. SOM může zákazníky rozdělit do podobných skupin, aniž bychom předem znali správné třídy.

### 4.3 Topologie

Kohonenova mapa má:

- vstupní vrstvu nebo vstupní terminály;
- jednu kompetiční vrstvu (competitive layer);
- neurony uspořádané do topologické struktury, často 1D řady nebo 2D mřížky.

Každý neuron v kompetiční vrstvě má váhový vektor stejné dimenze jako vstup:

```text
wj = [w1j, w2j, ..., wRj]^T
```

Tento váhový vektor lze chápat jako reprezentanta určité oblasti vstupního prostoru. Po naučení leží váhové vektory přibližně ve středech shluků nebo na místech, která reprezentují rozložení dat.

### 4.4 Vítězný neuron

Pro každý vstup `x` se hledá neuron, jehož váhový vektor je vstupu nejblíže. Tento neuron se nazývá vítězný neuron (winning neuron) nebo BMU (best matching unit).

Obvyklá vzdálenost:

```text
D(j) = Σ [xi - wij]^2
```

Vítěz:

```text
J = arg min D(j)
```

Vítězný neuron reprezentuje shluk, do kterého vstup nejlépe patří.

### 4.5 Okolí neuronu

Okolí neuronu (neighborhood) je množina neuronů blízkých vítězi v topologické mřížce. Důležité je, že blízkost se zde bere podle mapy, ne nutně podle vstupního prostoru.

Na začátku učení je okolí větší. To způsobí hrubé uspořádání celé mapy. Později se okolí zmenšuje, často až na samotného vítěze. Tím se mapa lokálně dolaďuje.

Toto je hlavní rozdíl proti jednoduchému shlukování typu k-means: SOM neupravuje jen jeden vítězný střed, ale i jeho topologické okolí, takže zachovává sousednostní strukturu.

### 4.6 Učení SOM

Postup učení:

1. Zvolí se počet neuronů nebo shluků.
2. Inicializují se váhy, obvykle náhodně.
3. Zvolí se počáteční rychlost učení `α`.
4. Zvolí se počáteční poloměr okolí.
5. Pro každý vstup se najde vítězný neuron.
6. Vítěz a jeho okolí se posunou směrem ke vstupu.
7. Postupně se zmenšuje `α` i okolí.

Aktualizace vah:

```text
wij(k) = wij(k-1) + α [xi(k) - wij(k-1)]
```

pro neurony v okolí vítěze.

Intuice:

- pokud neuron vyhraje, znamená to, že už je vstupu podobný;
- posunem směrem ke vstupu se stane ještě lepším reprezentantem;
- posun okolních neuronů zachová hladké uspořádání mapy;
- opakováním vznikne mapa, kde blízké neurony reprezentují podobná data.

### 4.7 Vybavování

Po naučení se nový vstup zařadí tak, že se znovu najde nejbližší neuron. Ten se aktivuje a ostatní zůstanou neaktivní.

Výstup SOM tedy není klasická číselná predikce, ale informace o příslušnosti ke shluku nebo pozici v mapě.

### 4.8 Omezení a praktické poznámky

Uživatel často musí předem zvolit počet neuronů nebo tvar mapy. To může výrazně ovlivnit výsledek. Příliš málo neuronů znamená hrubé shluky, příliš mnoho neuronů může vést k obtížné interpretaci.

Výsledek může záviset na inicializaci a pořadí dat, i když Kohonen uváděl relativní robustnost při dostatečně různých počátečních vahách.

SOM je velmi dobrá pro pochopení struktury dat, ale pokud potřebujeme přesnou klasifikaci do předem známých tříd, obvykle se používá učení s učitelem.

### 4.9 Co umět říct u zkoušky

Kohonenova mapa je samoorganizující síť učená bez učitele. Má kompetiční vrstvu, ve které neurony soutěží o vstup. Vítězný neuron a jeho okolí se během učení posouvají směrem ke vstupu. Výsledkem je mapa, kde podobná data aktivují stejné nebo blízké neurony. Používá se pro shlukovou analýzu, vizualizaci a rozpoznávání vzorů.

## 5. Dopředná vícevrstvá umělá neuronová síť

### 5.1 Co je dopředná vícevrstvá síť

Dopředná vícevrstvá umělá neuronová síť (multilayer feedforward neural network) je síť, ve které signál postupuje pouze jedním směrem: od vstupu přes skryté vrstvy k výstupu. Často se označuje jako vícevrstvý perceptron (multilayer perceptron, MLP), i když pojem MLP se v různých zdrojích používá mírně odlišně.

Základní struktura:

- vstupní uzly;
- jedna nebo více skrytých vrstev;
- výstupní vrstva;
- vážená spojení pouze mezi sousedními vrstvami;
- žádné zpětné vazby.

### 5.2 Proč je více vrstev důležitých

Jednoduchý perceptron umí pouze lineární hranici. Vícevrstvá síť dokáže skládat více nelineárních transformací a tím vytvářet složité rozhodovací hranice nebo aproximovat složité funkce.

Skryté vrstvy (hidden layers) vytvářejí vnitřní reprezentaci dat. V první vrstvě se mohou vytvořit jednoduché kombinace vstupů, v dalších vrstvách složitější znaky. U obrazových sítí to může být například hrana, roh, část objektu, celý objekt. U technických dat to může být skrytá nelineární kombinace měřených veličin.

### 5.3 K čemu slouží

Dopředné vícevrstvé sítě se používají jako:

- klasifikátory (classifiers);
- regresní modely (regression models);
- univerzální aproximátory (universal approximators);
- prediktory (predictors);
- modely systémů;
- rozhodovací členy;
- součásti řídicích systémů.

Ve skriptech je jim věnována velká pozornost, protože jsou obecně velmi použitelné.

### 5.4 Výpočet odezvy

Odezva sítě se počítá dopředným průchodem (forward pass). Pro neuron `j` ve vrstvě `k`:

```text
y_a,j^k = Σ w_ij^k y_i^(k-1)
y_j^k = φ^k(y_a,j^k)
```

Výstupy jedné vrstvy jsou vstupy vrstvy následující.

Postup:

1. Vstupní vektor se nastaví na vstupní uzly.
2. Spočítají se potenciály a výstupy první skryté vrstvy.
3. Stejně se spočítají další vrstvy.
4. Výstup poslední vrstvy je výstup celé sítě.

### 5.5 Učení sítě

Dopředná vícevrstvá síť se obvykle učí s učitelem. Máme trénovací data:

```text
(x, t)
```

Síť spočítá:

```text
y = F(x)
```

Chyba:

```text
e = t - y
```

Cílem učení je minimalizovat chybovou funkci (error function, loss function), například součet čtverců chyb:

```text
E = Σ (tj - yj)^2
```

Protože síť může mít mnoho vah, ruční nastavení není reálné. Používají se optimalizační algoritmy.

### 5.6 Zpětné šíření chyby

Zpětné šíření chyby (backpropagation) je základní algoritmus pro učení vícevrstvých dopředných sítí. Je založený na gradientním sestupu.

Hlavní myšlenka:

- nejprve spočítáme výstup sítě;
- zjistíme chybu na výstupu;
- určíme, jak za chybu odpovídají jednotlivé váhy;
- váhy upravíme ve směru, který chybu snižuje.

Pro váhu mezi neuronem `i` v předchozí vrstvě a neuronem `j` v aktuální vrstvě:

```text
Δw_ij^k = α δ_j^k y_i^(k-1)
```

kde:

- `α` je rychlost učení (learning rate);
- `δ_j^k` je lokální gradient (local gradient);
- `y_i^(k-1)` je výstup zdrojového neuronu.

Pro výstupní vrstvu:

```text
δ_j^C = e_j φ'(y_a,j^C)
```

Pro skrytou vrstvu:

```text
δ_j^k = φ'(y_a,j^k) Σ δ_l^(k+1) w_jl^(k+1)
```

Tento vztah ukazuje, proč se algoritmus jmenuje zpětné šíření chyby. Gradient ve skryté vrstvě závisí na gradientech vrstvy napravo, tedy chyba se počítá zpět od výstupu ke vstupu.

### 5.7 Diferencovatelnost aktivačních funkcí

Backpropagation potřebuje derivace aktivačních funkcí. Proto se u klasického výkladu používají spojité diferencovatelné funkce, například sigmoidální funkce, tanh nebo lineární funkce.

Skoková funkce se pro backpropagation nehodí, protože není diferencovatelná a neposkytuje použitelný gradient.

### 5.8 Rychlost učení a inicializace

Rychlost učení (learning rate) určuje velikost kroku při úpravě vah.

Příliš malá:

- učení je pomalé;
- může vyžadovat mnoho epoch.

Příliš velká:

- síť může přeskakovat minimum;
- chyba může oscilovat;
- učení může divergovat.

Počáteční inicializace vah (weight initialization) je důležitá, protože chybová funkce vícevrstvé sítě může mít mnoho lokálních minim. Ve skriptech je zdůrazněno, že algoritmus se často spouští opakovaně s různými počátečními vahami.

### 5.9 Online a offline učení

Online učení (online learning) aktualizuje váhy po každém vzoru. Může být rychlé a reaguje průběžně, ale bývá méně stabilní.

Offline nebo dávkové učení (offline learning, batch learning) akumuluje změny za více vzorů nebo celou epochu a potom váhy upraví. Bývá stabilnější, ale výpočetně náročnější.

### 5.10 Vylepšení učení

Momentum (momentum) přidává vliv předchozí změny vah. Pomáhá překonávat malé lokální nerovnosti chybové funkce a stabilizuje směr učení.

Adaptivní rychlost učení (adaptive learning rate) mění `α` podle úspěšnosti učení. Pokud chyba klesá, krok se může zvětšit. Pokud chyba roste, krok se zmenší.

RPROP (resilient backpropagation) používá hlavně znaménko gradientu, ne jeho velikost. To pomáhá v oblastech, kde jsou derivace velmi malé.

Kvazi-Newtonovy metody (quasi-Newton methods) a Levenberg-Marquardtův algoritmus (Levenberg-Marquardt algorithm) používají pokročilejší optimalizační principy. Často konvergují v menším počtu epoch, ale jednotlivé epochy jsou náročnější.

### 5.11 Volba topologie

Neexistuje univerzální analytický návod, jak zvolit počet vrstev a neuronů. Ve skriptech se uvádí empirické a experimentální postupy.

Možné strategie:

- začít jednoduchou sítí a přidávat neurony;
- začít větší sítí a zjednodušovat ji;
- porovnávat výsledky na validační množině;
- opakovat trénování pro různé inicializace;
- sledovat nejen trénovací chybu, ale hlavně validační a testovací chybu.

Praktická zásada: síť má být dost složitá na zachycení vztahu, ale ne zbytečně složitá. Příliš mnoho parametrů zvyšuje riziko přeučení.

### 5.12 Univerzální aproximátor

Dopředná vícevrstvá síť s vhodnou skrytou vrstvou může aproximovat libovolnou spojitou funkci s požadovanou přesností. Tomu se říká vlastnost univerzálního aproximátoru (universal approximation property).

Důležité je správné pochopení:

- věta říká, že taková síť existuje;
- neříká, jak snadno ji najdeme;
- neříká, že každé trénování skončí správně;
- neříká, že síť bude dobře generalizovat mimo rozsah dat.

Inženýrsky je proto klíčové mít kvalitní data, správnou normalizaci, vhodnou topologii a validaci.

### 5.13 Statické a dynamické modelování

Statický neuronový model (static neural model) popisuje ustálený vztah:

```text
yS = Ψ(u)
```

Používá se, když výstup závisí jen na aktuálních vstupech nebo na ustáleném stavu.

Dynamický neuronový model (dynamic neural model) popisuje chování v čase. Vstupem sítě jsou zpožděné hodnoty vstupů a výstupů:

```text
y(k) = Ψ[y(k-1), ..., y(k-n), u(k-1), ..., u(k-m)]
```

Tím dopředná síť získá informaci o historii systému, i když její struktura sama o sobě není rekurentní.

Při tvorbě dynamického modelu je nutné řešit:

- vzorkovací periodu;
- řád modelu;
- volbu vstupních zpoždění;
- kvalitu měřených dat;
- autonomní test modelu.

### 5.14 Co umět říct u zkoušky

Dopředná vícevrstvá síť je síť bez zpětných vazeb, kde signál postupuje od vstupu přes skryté vrstvy k výstupu. Díky nelineárním aktivačním funkcím dokáže řešit nelineární úlohy. Učí se nejčastěji s učitelem pomocí zpětného šíření chyby, které počítá gradient chyby od výstupní vrstvy zpět ke skrytým vrstvám. Síť může sloužit jako univerzální aproximátor, ale konkrétní topologie a parametry se v praxi hledají experimentálně.

## 6. Konvoluční neuronové sítě

### 6.1 Co je CNN

Konvoluční neuronová síť (convolutional neural network, CNN) je speciální typ neuronové sítě určený hlavně pro data s lokální a prostorovou strukturou. Nejtypičtější oblastí jsou obrazy, ale CNN se používají také pro video, zvuk, signály nebo některé časové řady.

CNN je v principu dopředná vícevrstvá síť, ale místo toho, aby každá vrstva byla plně propojená se všemi vstupy, používá lokální filtry.

### 6.2 Proč CNN vznikly

Obraz má mnoho vstupů. Například obrázek 224 x 224 x 3 má přes 150 tisíc hodnot. Plně propojená síť by měla obrovské množství vah. Navíc by ignorovala fakt, že sousední pixely spolu souvisejí.

CNN využívá tři důležité předpoklady:

- lokální struktura: blízké pixely spolu souvisí;
- sdílení vah: stejný příznak se může objevit kdekoliv v obraze;
- hierarchie příznaků: z hran vznikají tvary, z tvarů části objektů, z částí celé objekty.

### 6.3 Konvoluce

Konvoluce (convolution) v CNN znamená posouvání malého filtru přes vstup. Filtr nebo jádro (filter, kernel) je malá matice vah, například 3 x 3.

V každé poloze se spočítá vážený součet lokální oblasti. Výsledkem je mapa příznaků (feature map).

Intuice:

- filtr pro svislou hranu reaguje tam, kde je svislá hrana;
- filtr pro texturu reaguje tam, kde je daná textura;
- různé filtry vytvářejí různé mapy příznaků.

Filtry se neučí ručně. Jejich hodnoty jsou parametry sítě a učí se z dat pomocí backpropagation.

### 6.4 Lokální receptivní pole

Lokální receptivní pole (local receptive field) znamená, že neuron nesleduje celý obraz, ale jen malou oblast. To výrazně snižuje počet parametrů a odpovídá povaze obrazu, kde lokální vzory mají význam.

V hlubších vrstvách se efektivní receptivní pole zvětšuje. Neuron v hluboké vrstvě tak může reagovat na větší část obrazu, i když jednotlivé konvoluce jsou malé.

### 6.5 Sdílení vah

Sdílení vah (weight sharing) znamená, že stejný filtr se použije na mnoha pozicích obrazu. Jeden filtr tedy detekuje stejný typ příznaku kdekoliv.

Výhody:

- mnohem méně parametrů;
- lepší generalizace;
- schopnost detekovat příznak nezávisle na poloze;
- efektivní výpočet.

### 6.6 Aktivační funkce

Po konvoluci se používá nelineární aktivační funkce. V moderních CNN je nejběžnější ReLU:

```text
ReLU(x) = max(0, x)
```

Bez nelinearity by i konvoluční vrstvy zůstaly jen lineární transformací. Aktivační funkce umožňuje skládat složité příznaky.

### 6.7 Pooling

Poolingová vrstva (pooling layer) zmenšuje prostorový rozměr map příznaků.

Max pooling (max pooling):

```text
z oblasti vybere maximum
```

Average pooling (average pooling):

```text
z oblasti vezme průměr
```

Smysl poolingu:

- snižuje počet hodnot;
- snižuje výpočetní náročnost;
- zvyšuje robustnost vůči malým posunům;
- ponechává informaci, že příznak v dané oblasti existuje.

Moderní architektury někdy pooling nahrazují konvolucemi s krokem větším než 1, ale princip zmenšování reprezentace zůstává důležitý.

### 6.8 Typická architektura CNN

Jednoduchá CNN může mít tvar:

```text
vstupní obraz
-> konvoluce
-> ReLU
-> pooling
-> konvoluce
-> ReLU
-> pooling
-> zploštění nebo globální pooling
-> plně propojená vrstva
-> výstup
```

Pro klasifikaci se často používá softmax (softmax function), který převede výstupy na hodnoty interpretovatelné jako pravděpodobnosti tříd.

### 6.9 Učení CNN

CNN se učí stejně základním principem jako jiné dopředné sítě: pomocí zpětného šíření chyby. Rozdíl je v tom, že váhy filtrů jsou sdílené. Gradienty se tedy sčítají přes všechny pozice, kde byl filtr použit.

Při trénování klasifikátoru se používá trénovací množina obrázků a jejich tříd. Síť se postupně naučí filtry, které jsou užitečné pro minimalizaci chyby klasifikace.

### 6.10 Co se síť učí

Ve spodních vrstvách se typicky učí jednoduché příznaky:

- hrany;
- rohy;
- přechody jasu;
- jednoduché textury.

Ve středních vrstvách:

- části tvarů;
- vzory;
- struktury.

V hlubších vrstvách:

- části objektů;
- celé objekty;
- abstraktní reprezentace tříd.

To je hlavní rozdíl proti ručnímu návrhu příznaků. CNN si příznaky naučí sama.

### 6.11 Výhody a nevýhody CNN

Výhody:

- efektivní pro obrazová data;
- méně parametrů než plně propojená síť;
- učí se příznaky automaticky;
- dobrá robustnost vůči malým posunům;
- vysoká přesnost při dostatku dat.

Nevýhody:

- potřebuje hodně dat;
- může být výpočetně náročná;
- horší interpretovatelnost;
- citlivost na změnu domény dat;
- možnost přeučení.

### 6.12 Co umět říct u zkoušky

Konvoluční neuronová síť je dopředná síť specializovaná pro obrazová a prostorová data. Používá konvoluční filtry, které se posouvají přes vstup a vytvářejí mapy příznaků. Díky lokálním receptivním polím a sdílení vah má mnohem méně parametrů než plně propojená síť. Pooling snižuje rozměr a zvyšuje robustnost. CNN se učí pomocí backpropagation a její hlubší vrstvy reprezentují stále abstraktnější příznaky.

## 7. Porovnání hlavních typů sítí

| Síť | Anglicky | Typ učení | Typická úloha | Hlavní princip | Hlavní omezení |
|---|---|---|---|---|---|
| Jednoduchý perceptron | simple perceptron | s učitelem | binární klasifikace | lineární hranice | pouze lineárně separovatelná data |
| Hopfieldova síť | Hopfield network | Hebbovo učení | autoasociativní paměť | konvergence ke stabilnímu stavu | malá kapacita, fantomy, oscilace |
| Kohonenova mapa | self-organizing map, SOM | bez učitele | shlukování | vítěz a okolí se posouvají ke vstupu | volba počtu neuronů, interpretace |
| Dopředná vícevrstvá síť | multilayer feedforward network, MLP | s učitelem | aproximace, klasifikace, predikce | backpropagation přes vrstvy | volba topologie, lokální minima, přeučení |
| Konvoluční síť | convolutional neural network, CNN | s učitelem | obrazy, video | filtry, sdílení vah, pooling | data, výpočetní náročnost, interpretace |

## 8. Slovníček pojmů

- Umělá neuronová síť - artificial neural network, ANN
- Umělý neuron - artificial neuron
- Formální neuron - formal neuron
- McCulloch-Pittsův neuron - McCulloch-Pitts neuron
- Vstup - input
- Výstup - output
- Vstupní vektor - input vector
- Výstupní vektor - output vector
- Váha - weight
- Váha spojení - connection weight, synaptic weight
- Práh - threshold
- Bias - bias
- Agregační funkce - aggregation function
- Aktivační funkce - activation function, transfer function
- Vstupní potenciál - net input, activation potential
- Vrstva - layer
- Vstupní vrstva - input layer
- Skrytá vrstva - hidden layer
- Výstupní vrstva - output layer
- Topologie - topology, architecture
- Dopředná síť - feedforward network
- Rekurentní síť - recurrent network
- Zpětná vazba - feedback
- Učení - learning, training
- Učení s učitelem - supervised learning
- Učení bez učitele - unsupervised learning
- Kompetitivní učení - competitive learning
- Hebbovo učení - Hebbian learning
- Chybové učení - error learning
- Chybová funkce - error function, loss function
- Gradient - gradient
- Lokální gradient - local gradient
- Rychlost učení - learning rate
- Epocha - epoch
- Trénovací množina - training set
- Validační množina - validation set
- Testovací množina - test set
- Generalizace - generalization
- Přeučení - overfitting
- Podučení - underfitting
- Normalizace - normalization
- Lineární separovatelnost - linear separability
- Perceptron - perceptron
- Vícevrstvý perceptron - multilayer perceptron, MLP
- Zpětné šíření chyby - backpropagation
- Momentum - momentum
- Lokální minimum - local minimum
- Univerzální aproximátor - universal approximator
- Autoasociativní paměť - autoassociative memory
- Stabilní stav - stable state
- Atraktor - attractor
- Fantomový stav - spurious state
- Hammingova vzdálenost - Hamming distance
- Shluk - cluster
- Shluková analýza - cluster analysis
- Samoorganizace - self-organization
- Samoorganizační mapa - self-organizing map, SOM
- Vítězný neuron - winning neuron, best matching unit, BMU
- Okolí neuronu - neighborhood
- Konvoluce - convolution
- Konvoluční vrstva - convolutional layer
- Filtr, jádro - filter, kernel
- Mapa příznaků - feature map
- Lokální receptivní pole - local receptive field
- Sdílení vah - weight sharing
- Pooling - pooling
- Max pooling - max pooling
- Average pooling - average pooling
- Plně propojená vrstva - fully connected layer
- ReLU - rectified linear unit
- Softmax - softmax

## 9. Zkouškové otázky s odpověďmi

### Co je umělá neuronová síť?

Umělá neuronová síť je soustava propojených umělých neuronů, která realizuje transformační funkci mezi vstupy a výstupy. Její chování je dáno topologií, aktivačními funkcemi a hodnotami vah. Učením se nastavují váhy tak, aby síť řešila požadovanou úlohu.

### Proč jsou důležité aktivační funkce?

Aktivační funkce převádí vstupní potenciál neuronu na výstup. Zavádí nelinearitu. Bez nelinearity by i vícevrstvá síť byla pouze lineární transformací a neuměla by modelovat složité nelineární vztahy.

### Jaký je rozdíl mezi učením s učitelem a bez učitele?

Učení s učitelem pracuje s dvojicemi vstup a cílový výstup. Síť minimalizuje chybu mezi skutečným a požadovaným výstupem. Učení bez učitele nemá cílové výstupy a hledá strukturu ve vstupních datech, například shluky.

### Proč jednoduchý perceptron neumí XOR?

Protože XOR není lineárně separovatelná úloha. Jednoduchý perceptron vytváří pouze jednu lineární rozhodovací hranici. XOR vyžaduje nelineární oddělení tříd, které zvládne až vícevrstvá síť.

### Jak funguje Hopfieldova síť jako paměť?

Vzory jsou uloženy do matice vah Hebbovým pravidlem. Při vybavování dostane síť počáteční, třeba poškozený vzor, a iterativně aktualizuje stavy neuronů. Pokud je vstup dost blízko uloženému vzoru, síť konverguje k jeho stabilní reprezentaci.

### Jak funguje Kohonenova mapa?

Pro každý vstup se najde neuron s nejbližším váhovým vektorem. Tento vítězný neuron a jeho okolí se posunou směrem ke vstupu. Opakováním vznikne mapa, která zachovává podobnost vstupních dat a rozděluje je do shluků.

### Co je backpropagation?

Backpropagation je algoritmus učení vícevrstvé dopředné sítě. Nejprve se spočítá výstup sítě, potom chyba na výstupu, následně se vypočtou lokální gradienty od výstupní vrstvy zpět ke skrytým vrstvám a podle nich se upraví váhy.

### Co znamená univerzální aproximátor?

Znamená to, že dopředná vícevrstvá síť s vhodnou strukturou a nelineární aktivační funkcí může aproximovat spojitou funkci s libovolně požadovanou přesností. V praxi ale musíme vhodnou síť najít experimentálně a ověřit její generalizaci.

### Proč jsou CNN vhodné pro obraz?

Protože obraz má lokální prostorovou strukturu. CNN používá lokální filtry, sdílení vah a pooling. Tím snižuje počet parametrů, zachovává prostorovou informaci a učí se příznaky od jednoduchých hran až po složité objekty.

## 10. Nejkratší opakovací mapa

Neuron = vážený součet + bias + aktivační funkce.  
Síť = propojené neurony uspořádané do topologie.  
Učení = změna vah podle dat nebo podobnosti.  
Perceptron = lineární binární klasifikátor.  
Hopfield = rekurentní autoasociativní paměť.  
Kohonen = samoorganizace a shlukování bez učitele.  
MLP/DVS = nelineární aproximace pomocí vrstev a backpropagation.  
CNN = konvoluční filtry, sdílení vah a pooling pro obrazová data.
