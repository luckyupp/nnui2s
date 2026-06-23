# Výtah ke zkoušce: umělé neuronové sítě

Zdroj: skripta `Úvod do umělých neuronových sítí (1).pdf`, doplněno o samostatnou část ke konvolučním neuronovým sítím, protože ve skriptech jsou CNN pouze zmíněny mezi dalšími topologiemi.

## Jak se na to dívat celkově

Umělá neuronová síť (artificial neural network, ANN) je výpočetní model složený z jednoduchých výpočetních jednotek - umělých neuronů (artificial neurons). Síť se nesnaží řešit úlohu pomocí ručně napsaných pravidel typu "jestliže - pak", ale naučí se vztah mezi vstupy a výstupy z dat.

Typická intuice:

- vstupy jsou měřené vlastnosti, pixely, signály, historické hodnoty apod.;
- váhy (weights) říkají, jak důležité jsou jednotlivé vstupy;
- neuron spočítá vážený součet a přes aktivační funkci vytvoří výstup;
- síť jako celek realizuje transformační funkci: převádí vstupní vektor na výstupní vektor;
- učení (learning/training) znamená změnu vah a prahů tak, aby výstup sítě odpovídal požadovanému chování.

Příklad: místo ručního pravidla "pokud je lupínek dost těžký a dost světlý, expeduj ho" se síť naučí rozhodovací hranici z příkladů lupínků označených jako vyhověl/nevyhověl.

## 1. Umělé neuronové sítě, členění, definice pojmů

### Základní pojmy

Umělý neuron (artificial neuron) je matematický model inspirovaný biologickým neuronem. Má vstupy, váhy, agregační funkci, aktivační funkci a výstup.

Vstup (input) je hodnota přiváděná do neuronu. Může být:

- kvalitativní, například 0/1 pro "vlastnost platí/neplatí";
- kvantitativní, například teplota, hmotnost, intenzita signálu;
- výstup z předchozího neuronu.

Vstupní vektor (input vector) je sada vstupů:

```text
x = [x1, x2, ..., xR]
```

Váha spojení (connection weight, synaptic weight) určuje sílu vlivu konkrétního vstupu. Velká kladná váha vstup podporuje, záporná váha ho tlumí, váha blízko nule znamená malý vliv.

Práh neuronu (threshold) nebo bias (bias) posouvá rozhodování neuronu. Prakticky se často modeluje jako zvláštní vstup `x0 = 1` s váhou `w0`.

Agregační funkce (aggregation function) sloučí vstupy do jedné hodnoty. V základním modelu jde o vážený součet:

```text
ya = Σ wi xi
```

Vstupní potenciál (net input, activation potential) `ya` je hodnota před aktivační funkcí.

Aktivační funkce (activation function, transfer function) převede vstupní potenciál na výstup neuronu:

```text
y = φ(ya)
```

Nejčastější aktivační funkce:

- skoková funkce (step function): výstup je například -1/1 nebo 0/1;
- lineární funkce (linear function): výstup odpovídá vstupnímu potenciálu;
- saturovaná lineární funkce (saturated linear function): lineární jen v určitém rozsahu;
- sigmoidální funkce (sigmoid function): výstup obvykle v intervalu 0 až 1;
- hyperbolický tangens (hyperbolic tangent, tanh): výstup v intervalu -1 až 1;
- Gaussova funkce (Gaussian function): používá se například u radiálních sítí.

Důležité pochopení: bez nelineární aktivační funkce by více vrstev lineárních neuronů pořád dělalo jen jednu lineární transformaci. Nelinearita dává síti schopnost modelovat složité závislosti.

### Topologie a členění sítí

Topologie sítě (network topology/architecture) znamená rozmístění neuronů a jejich propojení.

Vrstva (layer):

- vstupní vrstva (input layer): přivádí data do sítě, často se nepočítá jako aktivní neuronová vrstva;
- skrytá vrstva (hidden layer): mezivrstva, kde síť vytváří vnitřní reprezentaci problému;
- výstupní vrstva (output layer): dává výslednou odpověď.

Dopředná síť (feedforward neural network) má signál veden jen jedním směrem, od vstupu k výstupu. Nemá zpětné vazby.

Rekurentní nebo zpětnovazební síť (recurrent neural network, feedback network) má zpětné vazby. Výstup může záviset nejen na aktuálním vstupu, ale i na předchozím stavu sítě. Typický příklad ve skriptech je Hopfieldova síť.

Učení s učitelem (supervised learning) znamená, že máme dvojice vstup - správný výstup. Síť porovnává svůj výstup s cílem a upravuje váhy podle chyby.

Učení bez učitele (unsupervised learning) znamená, že máme jen vstupy bez správných odpovědí. Síť hledá strukturu v datech, například shluky.

Chybová funkce (error/loss function) měří rozdíl mezi požadovaným a skutečným výstupem. Častý tvar:

```text
E = Σ (t - y)^2
```

kde `t` je cílová hodnota (target) a `y` je skutečný výstup sítě.

Epocha (epoch) je jedno projití celé trénovací množiny.

Trénovací množina (training set) slouží k úpravě vah. Validační množina (validation set) slouží k výběru vhodné sítě a zastavení učení. Testovací množina (test set) slouží až na konečné ověření kvality.

Normalizace dat (data normalization) znamená převod dat do vhodného rozsahu, často `<-1; 1>` nebo `<0; 1>`. Je důležitá, protože sítě se typicky učí stabilněji, když vstupy nejsou v úplně odlišných měřítkách.

### Hebbův zákon učení

Hebbovo učení (Hebbian learning) lze intuitivně shrnout: když jsou dva propojené neurony aktivní současně, jejich spojení se posiluje. Často se popisuje větou "neurons that fire together wire together".

Ve skriptech je Hebbův zákon důležitý hlavně jako historický základ učení a jako pravidlo využitelné u jednoduchého perceptronu a Hopfieldovy sítě.

## 2. Neuronová síť typu jednoduchý perceptron

Jednoduchý perceptron (single-layer perceptron/simple perceptron) je jeden výpočetní neuron používaný hlavně pro binární klasifikaci.

### K čemu slouží

Perceptron rozhoduje, do které ze dvou tříd vstup patří. Například:

- vyhověl/nevyhověl;
- ano/ne;
- třída A/třída B.

Jeho zásadní omezení: umí řešit pouze lineárně separovatelné problémy (linearly separable problems). To znamená, že třídy musí jít oddělit přímkou, rovinou nebo obecně nadrovinou.

Typický zkouškový příklad: jednoduchý perceptron nezvládne XOR, protože XOR není lineárně separovatelný.

### Jak funguje

Perceptron spočítá vážený součet vstupů:

```text
ya = Σ wi xi
```

nebo maticově:

```text
ya = wT x
```

Potom použije aktivační funkci:

```text
y = φ(ya)
```

U bipolární skokové funkce:

```text
y =  1, pokud ya >= 0
y = -1, pokud ya < 0
```

Intuice: perceptron vytvoří rozhodovací hranici. Na jedné straně hranice dává výstup 1, na druhé straně -1.

### Učení perceptronu

Perceptron se učí s učitelem (supervised learning). Máme trénovací dvojice `x` a `t`, kde `t` je požadovaný výstup.

Chyba:

```text
e = t - y
```

Úprava vah:

```text
Δwi = α xi e
wi = wi + Δwi
```

kde `α` je koeficient rychlosti učení (learning rate).

Prakticky:

1. nastaví se počáteční váhy, často malá náhodná čísla;
2. síti se předloží vzor;
3. spočítá se výstup;
4. spočítá se chyba;
5. váhy se posunou tak, aby příště byl výstup blíž cíli;
6. opakuje se do splnění ukončovací podmínky.

Důležité: pokud data nejsou lineárně separovatelná, perceptron nemusí najít dokonalé řešení. Může ale najít použitelné přibližné rozhodování.

## 3. Hopfieldova síť

Hopfieldova síť (Hopfield network) je rekurentní síť s jednou vrstvou, kde je každý neuron propojen s každým jiným neuronem, ale ne sám se sebou.

### K čemu slouží

Hopfieldova síť se často vysvětluje jako autoasociativní paměť (autoassociative memory).

To znamená: síť si zapamatuje několik vzorů. Když jí dáme poškozený nebo neúplný vzor, měla by se postupně ustálit na nejbližším uloženém vzoru.

Příklad:

- uložíme několik binárních obrázků;
- síti předložíme zašuměnou verzi jednoho obrázku;
- síť iteracemi "dopočítá" původní vzor.

### Jak funguje

Síť má `R` neuronů a stav celé sítě je vektor hodnot, typicky bipolární:

```text
y_i ∈ {-1, 1}
```

Váhy jsou obvykle symetrické:

```text
wij = wji
```

a na diagonále jsou nuly:

```text
wii = 0
```

Výstup neuronu závisí na výstupech ostatních neuronů z předchozího kroku. Proto jde o rekurentní síť (recurrent network).

Vybavování (recall/retrieval) probíhá iterativně:

1. síti se dá počáteční stav, například poškozený vzor;
2. neurony se aktualizují podle vah a aktivační funkce;
3. síť se opakovaně přepočítává;
4. skončí, když se stav už nemění, nebo když osciluje.

Synchronní aktualizace (synchronous update): aktualizují se všechny neurony najednou.

Asynchronní aktualizace (asynchronous update): aktualizuje se vždy jeden neuron, často v náhodném pořadí.

### Učení Hopfieldovy sítě

Učení je jednorázové, podle Hebbova principu. Pro uložené vzory `t(p)` se váhy nastaví přibližně jako součet součinů odpovídajících prvků vzorů:

```text
wij = Σ ti(p) tj(p), pro i ≠ j
wij = 0, pro i = j
```

Intuice: pokud mají dva prvky ve vzorech často stejnou hodnotu, vazba mezi nimi se posílí.

### Omezení

Kapacita Hopfieldovy sítě je omezená. Ve skriptech je uveden přibližný vztah:

```text
N < 0,138 R
```

kde `N` je počet zapamatovatelných vzorů a `R` počet neuronů.

Další problémy:

- fantomový stav (spurious state): síť skončí ve vzoru, který nebyl trénovací;
- oscilace (oscillation): síť střídá dva stavy;
- podobné vzory se mohou plést;
- výsledek není vždy jistý.

Zkouškově důležité: Hopfieldova síť není klasifikátor jako perceptron, ale paměťový model, který hledá stabilní stav nejbližší vstupu.

## 4. Kohonenova samoorganizační mapa

Kohonenova samoorganizační mapa (Kohonen self-organizing map, SOM) je síť učená bez učitele. Používá se hlavně pro shlukovou analýzu (cluster analysis) a vizualizaci struktury dat.

### K čemu slouží

SOM hledá podobnosti ve vstupních datech. Vstupy, které jsou si podobné, přiřazuje k blízkým neuronům v mapě.

Typické použití:

- shlukování dat;
- rozpoznávání vzorů;
- zjednodušení velkého množství dat;
- vizualizace vícerozměrných dat na 2D mapě;
- rozpoznávání číslic nebo typických tvarů.

### Jak vypadá

Síť má vstupní terminály a jednu kompetiční vrstvu (competitive layer). Neurony v této vrstvě jsou často uspořádány do mřížky.

Kompetitivní učení (competitive learning): pro každý vstup "soutěží" neurony o to, který mu je nejblíže. Vítězí jeden neuron.

Vítězný neuron se často označuje jako BMU (best matching unit), ve skriptech jako neuron `J`.

### Jak funguje učení

Každý neuron má váhový vektor. Tento vektor lze chápat jako pozici neuronu v prostoru vstupních dat.

Pro každý vstup:

1. spočítá se vzdálenost vstupu od všech váhových vektorů;
2. vybere se vítězný neuron s nejmenší vzdáleností;
3. upraví se váhy vítěze a jeho okolí směrem ke vstupu;
4. postupně se zmenšuje rychlost učení i velikost okolí.

Typická vzdálenost:

```text
D(j) = Σ [wij - xi]^2
```

Aktualizace vah:

```text
wij(k) = wij(k-1) + α [xi(k) - wij(k-1)]
```

pro vítězný neuron a neurony v jeho okolí.

Okolí neuronu (neighborhood) je množina neuronů blízkých vítězi v topologické mřížce. Na začátku bývá okolí větší, aby se mapa hrubě uspořádala. Později se zmenšuje, aby se doladily detaily.

### Vybavování SOM

Po naučení síti předložíme nový vstup. Spočítají se vzdálenosti k neuronům a aktivuje se nejbližší neuron. Ten určí shluk, do kterého vstup patří.

Zkouškově důležité:

- SOM nemá požadovaný výstup `t`;
- uživatel často předem volí počet shluků/neuronů;
- SOM zachovává topologii: podobné vstupy mají aktivovat blízké neurony;
- výstup je typicky "tento vstup patří k tomuto shluku".

## 5. Dopředná vícevrstvá umělá neuronová síť

Dopředná vícevrstvá síť (multilayer feedforward neural network) je síť, kde signál teče od vstupů přes jednu nebo více skrytých vrstev k výstupu.

Často se označuje jako vícevrstvý perceptron (multilayer perceptron, MLP), i když přesně vzato se může lišit použitými aktivačními funkcemi.

### K čemu slouží

Je to univerzální typ sítě pro:

- klasifikaci;
- regresi;
- aproximaci funkcí;
- modelování systémů;
- predikci;
- rozpoznávání vzorů.

Ve skriptech je zdůrazněno, že dopředná vícevrstvá síť může fungovat jako univerzální aproximátor (universal approximator). To znamená, že při vhodné topologii a vhodném učení dokáže s požadovanou přesností aproximovat širokou třídu spojitých funkcí.

### Topologie

Síť se skládá z:

- vstupních uzlů;
- jedné nebo více skrytých vrstev;
- výstupní vrstvy.

Spojení jsou:

- úplná mezi sousedními vrstvami;
- jednosměrná;
- bez zpětné vazby;
- bez spojů uvnitř stejné vrstvy.

Pro jeden neuron ve vrstvě:

```text
yk_aj = Σ wk_ij y(k-1)_i
yk_j = φk(yk_aj)
```

Odezva sítě se počítá postupně dopředu:

1. vstupní vektor se přivede na vstup;
2. spočítají se výstupy první skryté vrstvy;
3. ty se stanou vstupem další vrstvy;
4. pokračuje se až do výstupní vrstvy.

Tomu se říká dopředný průchod (forward pass).

### Učení

Dopředná vícevrstvá síť se obvykle učí s učitelem (supervised learning). Máme vstup `x`, cílový výstup `t`, síť spočítá výstup `y`, chyba je:

```text
e = t - y
```

Hlavní algoritmus je zpětné šíření chyby (backpropagation).

### Zpětné šíření chyby

Backpropagation znamená:

1. provedeme dopředný průchod a získáme výstup;
2. spočítáme chybu na výstupu;
3. chybu rozpočítáme zpět do předchozích vrstev;
4. podle lokálních gradientů upravíme váhy.

Lokální gradient (local gradient) říká, jak moc daný neuron přispěl k chybě a jakým směrem se mají měnit jeho váhy.

Úprava váhy má tvar:

```text
Δwk_ij = α δk_j y(k-1)_i
```

kde:

- `α` je rychlost učení (learning rate);
- `δk_j` je lokální gradient cílového neuronu;
- `y(k-1)_i` je výstup zdrojového neuronu z předchozí vrstvy.

Pro výstupní vrstvu se lokální gradient počítá z chyby na výstupu. Pro skrytou vrstvu se chyba odvozuje z gradientů vrstvy napravo. Proto se chyba šíří zpět, proti směru výpočtu výstupu.

Backpropagation vyžaduje diferencovatelné aktivační funkce (differentiable activation functions), typicky sigmoid, tanh nebo lineární funkci.

### Volba topologie a učení

Neexistuje přesný obecný návod, kolik vrstev a neuronů zvolit. Postupuje se experimentálně:

- začít jednoduše a postupně síť zvětšovat;
- nebo začít větší sítí a postupně ji zjednodušovat;
- používat validační množinu;
- opakovat trénování s různými počátečními vahami.

Počáteční inicializace vah (weight initialization) je důležitá, protože algoritmus může skončit v lokálním minimu (local minimum).

Rychlost učení (learning rate) `α`:

- příliš malá: učení je pomalé;
- příliš velká: algoritmus může minimum přeskakovat nebo divergovat;
- ve skriptech se pro data v rozsahu `<-1; 1>` doporučuje často zkoušet hodnoty přibližně `<0,01; 0,1>`.

Další algoritmy a vylepšení:

- momentum (momentum): bere v úvahu i předchozí změnu vah;
- adaptivní rychlost učení (adaptive learning rate);
- RPROP (resilient backpropagation);
- kvazi-Newtonovy metody (quasi-Newton methods);
- Levenberg-Marquardtův algoritmus (Levenberg-Marquardt algorithm).

### Univerzální aproximátor

Pojem univerzální aproximátor (universal approximator) znamená, že dopředná vícevrstvá síť s vhodnou skrytou vrstvou může aproximovat libovolnou spojitou funkci s požadovanou přesností.

Důležité dovysvětlení: tato věta říká, že taková síť existuje, ale neříká automaticky, jak ji najít. V praxi se musí hledat vhodná topologie, vhodné učení a vhodná data.

### Modelování a predikce

Statický model (static model) popisuje ustálenou závislost:

```text
yS = Ψ(u)
```

Síť se učí převod vstupů systému na ustálený výstup.

Dynamický model (dynamic model) popisuje chování v čase. Výstup závisí na minulých výstupech a minulých vstupech:

```text
y(k) = Ψ[y(k-1), ..., y(k-n), u(k-1), ..., u(k-m)]
```

Intuice: síť dostane "paměť" tím, že jí jako vstupy dáme zpožděné hodnoty. Samotná síť může být pořád dopředná, ale vstupní data obsahují historii.

## 6. Konvoluční neuronové sítě

Konvoluční neuronová síť (convolutional neural network, CNN) je speciální typ neuronové sítě určený hlavně pro data s prostorovou strukturou, typicky obrázky.

Ve skriptech nejsou CNN samostatně vysvětlené, ale ke zkoušce je potřeba chápat jejich princip.

### K čemu slouží

CNN se používají hlavně pro:

- rozpoznávání obrazu (image recognition);
- klasifikaci obrázků (image classification);
- detekci objektů (object detection);
- segmentaci obrazu (image segmentation);
- zpracování videa;
- někdy také pro signály, zvuk nebo časové řady.

Příklad: místo aby síť dostala ručně měřené vlastnosti obrázku, učí se sama detekovat hrany, rohy, textury, části objektů a nakonec celé objekty.

### Proč nestačí obyčejná plně propojená síť

U obrázku 100 x 100 pixelů máme 10 000 vstupů, u barevného obrázku třikrát tolik. Kdyby byl každý pixel napojen na každý neuron ve velké skryté vrstvě, počet vah by byl obrovský.

Navíc obraz má lokální strukturu: sousední pixely spolu souvisí. Běžná plně propojená síť tuto lokální strukturu nevyužívá efektivně.

CNN používá:

- lokální receptivní pole (local receptive field): neuron sleduje jen malou část vstupu;
- sdílení vah (weight sharing): stejný filtr se používá na různých místech obrazu;
- konvoluci (convolution): posouvání filtru přes obraz;
- pooling: zmenšení prostorového rozměru a zvýšení odolnosti vůči malým posunům.

### Konvoluční vrstva

Konvoluční vrstva (convolutional layer) používá filtr/jádro (filter/kernel). Filtr je malá matice vah, například 3 x 3 nebo 5 x 5, která se posouvá přes obraz.

V každé poloze filtr spočítá vážený součet hodnot v lokálním okolí. Výsledkem je mapa příznaků (feature map).

Intuice:

- jeden filtr může reagovat na svislé hrany;
- jiný na vodorovné hrany;
- další na rohy nebo textury;
- hlubší vrstvy skládají jednoduché příznaky do složitějších.

Mapa příznaků (feature map) říká, kde se v obraze daný příznak nachází.

### Aktivační funkce v CNN

Po konvoluci se používá nelineární aktivační funkce. Dnes nejčastěji ReLU:

```text
ReLU(x) = max(0, x)
```

ReLU (rectified linear unit) je jednoduchá a pomáhá rychlejšímu učení hlubokých sítí.

### Pooling

Poolingová vrstva (pooling layer) zmenšuje rozměr mapy příznaků.

Max pooling (max pooling) vezme z malého okolí maximum, například z oblasti 2 x 2.

Average pooling (average pooling) vezme průměr.

Smysl:

- sníží počet hodnot;
- zmenší výpočetní náročnost;
- zvýší odolnost vůči malému posunu objektu;
- ponechá informaci, že určitý příznak v oblasti existuje.

### Plně propojená část

Po několika konvolučních a poolingových vrstvách se často použije plně propojená vrstva (fully connected layer), která z naučených příznaků udělá finální rozhodnutí.

Například:

```text
obraz -> konvoluce -> ReLU -> pooling -> konvoluce -> ReLU -> pooling -> plně propojená vrstva -> softmax
```

Softmax (softmax function) se často používá ve výstupní vrstvě pro vícetřídní klasifikaci. Převede výstupy na hodnoty interpretovatelné jako pravděpodobnosti tříd.

### Co je na CNN zásadní

CNN se neučí jen klasifikátor, ale i příznaky. Starší přístup by byl: člověk ručně navrhne detektory hran, textur nebo tvarů a klasifikátor pak rozhodne. CNN si vhodné filtry naučí sama z dat.

Zkouškově důležité:

- konvoluce = filtr se posouvá přes vstup;
- sdílení vah výrazně snižuje počet parametrů;
- lokální propojení využívá prostorovou strukturu obrazu;
- pooling snižuje rozměr a zvyšuje robustnost;
- hlubší vrstvy reprezentují abstraktnější příznaky;
- CNN se typicky učí zpětným šířením chyby stejně jako jiné hluboké sítě.

## Rychlé porovnání hlavních sítí

| Síť | Anglicky | Učení | Hlavní použití | Klíčová myšlenka |
|---|---|---|---|---|
| Jednoduchý perceptron | simple perceptron | s učitelem | binární klasifikace | lineární rozhodovací hranice |
| Hopfieldova síť | Hopfield network | Hebbovo, jednorázové | autoasociativní paměť | konvergence ke stabilnímu uloženému vzoru |
| Kohonenova mapa | self-organizing map, SOM | bez učitele | shlukování, vizualizace | vítězný neuron a jeho okolí se posouvají ke vstupu |
| Dopředná vícevrstvá síť | multilayer feedforward network, MLP | s učitelem, backpropagation | klasifikace, regrese, aproximace | vrstvy nelineárních neuronů modelují složité funkce |
| Konvoluční síť | convolutional neural network, CNN | s učitelem, backpropagation | obraz, video, prostorová data | filtry detekují lokální příznaky sdílenými vahami |

## Slovníček důležitých termínů

- Umělá neuronová síť - artificial neural network, ANN
- Umělý neuron - artificial neuron
- Formální neuron - formal neuron
- Vstup - input
- Výstup - output
- Vstupní vektor - input vector
- Váha - weight
- Váha spojení - connection weight / synaptic weight
- Práh - threshold
- Bias - bias
- Agregační funkce - aggregation function
- Aktivační funkce - activation function / transfer function
- Vstupní potenciál - net input / activation potential
- Topologie sítě - network topology / architecture
- Vrstva - layer
- Vstupní vrstva - input layer
- Skrytá vrstva - hidden layer
- Výstupní vrstva - output layer
- Dopředná síť - feedforward network
- Rekurentní síť - recurrent network
- Zpětná vazba - feedback
- Učení - learning / training
- Učení s učitelem - supervised learning
- Učení bez učitele - unsupervised learning
- Kompetitivní učení - competitive learning
- Hebbovo učení - Hebbian learning
- Chybové učení - error learning
- Chybová funkce - error function / loss function
- Cílová hodnota - target
- Epocha - epoch
- Rychlost učení - learning rate
- Trénovací množina - training set
- Validační množina - validation set
- Testovací množina - test set
- Normalizace - normalization
- Lineární separovatelnost - linear separability
- Perceptron - perceptron
- Vícevrstvý perceptron - multilayer perceptron, MLP
- Zpětné šíření chyby - backpropagation
- Gradient - gradient
- Lokální gradient - local gradient
- Lokální minimum - local minimum
- Univerzální aproximátor - universal approximator
- Autoasociativní paměť - autoassociative memory
- Hammingova vzdálenost - Hamming distance
- Shluk - cluster
- Shluková analýza - cluster analysis
- Samoorganizace - self-organization
- Samoorganizační mapa - self-organizing map, SOM
- Vítězný neuron - winning neuron / best matching unit, BMU
- Okolí neuronu - neighborhood
- Konvoluce - convolution
- Konvoluční vrstva - convolutional layer
- Filtr / jádro - filter / kernel
- Mapa příznaků - feature map
- Sdílení vah - weight sharing
- Pooling - pooling
- Max pooling - max pooling
- Plně propojená vrstva - fully connected layer
- ReLU - rectified linear unit
- Softmax - softmax

## Typické zkouškové otázky a krátké odpovědi

**Co je umělá neuronová síť?**  
Síť propojených umělých neuronů, která realizuje transformační funkci ze vstupů na výstupy a učí se ji úpravou vah.

**Co je aktivační funkce?**  
Funkce, která převádí vstupní potenciál neuronu na jeho výstup. Anglicky activation function. Dává síti nelinearitu.

**Jaký je rozdíl mezi učením s učitelem a bez učitele?**  
Učení s učitelem má cílové výstupy, učení bez učitele hledá strukturu pouze ve vstupních datech.

**Proč jednoduchý perceptron neumí XOR?**  
Protože XOR není lineárně separovatelný. Jednoduchý perceptron vytváří jen jednu lineární rozhodovací hranici.

**K čemu je Hopfieldova síť?**  
K autoasociativní paměti: z neúplného nebo zašuměného vstupu se snaží vybavit nejbližší uložený vzor.

**K čemu je Kohonenova mapa?**  
Ke shlukování a uspořádání dat bez učitele. Podobné vstupy aktivují stejné nebo blízké neurony.

**Co je backpropagation?**  
Algoritmus učení vícevrstvé dopředné sítě, kde se chyba z výstupu šíří zpět do skrytých vrstev a podle gradientů se upravují váhy.

**Co znamená univerzální aproximátor?**  
Dopředná vícevrstvá síť s vhodnou strukturou dokáže aproximovat spojitou funkci s požadovanou přesností, ale konkrétní síť se musí najít experimentálně.

**Proč CNN fungují dobře na obrazech?**  
Protože využívají lokální strukturu obrazu, sdílené filtry a pooling. Učí se příznaky přímo z dat.

## Nejkratší mentální mapa

Neuron = vážený součet + bias + aktivační funkce.  
Síť = mnoho neuronů propojených do topologie.  
Učení = úprava vah podle dat.  
Perceptron = jednoduchá lineární klasifikace.  
Hopfield = paměť a stabilní stavy.  
Kohonen = shlukování bez učitele.  
MLP/DVS = obecná dopředná síť, backpropagation, aproximace funkcí.  
CNN = konvoluční filtry pro obraz a prostorová data.
