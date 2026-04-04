#set text(lang: "it")
// --- INTESTAZIONE: STILE ELEGANTE ---
#set page(
  paper: "a4", 
  margin: 2.3cm,
  numbering: "1" // Aggiunge i numeri di pagina in basso al centro
)
#set text(font: "Linux Libertine", lang: "it", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1.") // Numerazione romana per i capitoli principali

// Stile blocchi di codice (con barra laterale invece di bordo completo)
#show raw: set text(font: "Cascadia Code", size: 9.5pt)
#show raw.where(block: true): it => block(
  fill: luma(250),
  stroke: (left: 3pt + rgb("#555555")), // Barra grigia a sinistra
  inset: (left: 10pt, top: 8pt, bottom: 8pt, right: 8pt),
  width: 100%,
  it
)

/* // Stile blocchi di codice (barra laterale + numeri di riga)
#show raw.where(block: true): it => block(
  fill: luma(250),
  stroke: (left: 3pt + rgb("#555555")), // Barra grigia a sinistra
  inset: (left: 10pt, top: 8pt, bottom: 8pt, right: 8pt),
  width: 100%,
  {
    // Estrae le righe di codice e crea una griglia a due colonne
    let lines = it.text.split("\n")
    grid(
      columns: (auto, 1fr),
      column-gutter: 1.5em,
      ..lines.enumerate().map(((i, line)) => (
        // Colonna 1: Numero di riga (in grigio chiaro)
        align(right)[#text(fill: luma(160), size: 8.5pt, str(i + 1))],
        // Colonna 2: Il codice effettivo
        raw(line, lang: it.lang)
      )).flatten()
    )
  }
) */

// Didascalie
#show figure.caption: it => {
  // Imposta la dimensione base della caption
  set text(size: 10pt) 
  
  // Applica una riduzione solo al testo "raw" (monospace) dentro la caption
  // 0.85em di solito è il "magic number" per far sembrare il Cascadia grande quanto il testo normale
  show raw: set text(size: 0.85em) 
  
  it
}
// Funzione per il titolo (con autore e data integrati)
#let title(body) = align(center)[
  #block(above: 2em, below: 1em)[
    #text(size: 1.5em, weight: "bold", font: "Linux Biolinum", body)
  ]
  #block(below: 3em, above: 2em)[
    #text(size: 1.1em, "Diego Quarantani") \
    #v(0.3em)
    #text(size: 1em, "Aprile 2026") // Puoi modificare la data a piacimento
  ]
]
// --- FINE INTESTAZIONE ---

#title("Relazione – Laboratori di Data Science (Discesa del gradiente e Regressione Polinomiale)")

= Design del codice

== Generazione dei dati

```py
noise_amp = 0.2
n_samples = 200
eta = 0.5

x = np.random.uniform(0, 1, n_samples)
noise = np.random.uniform(-noise_amp, noise_amp, n_samples)
y = np.sin(2 * np.pi * x) + noise

x_true = np.linspace(0, 1, 500)
y_true = np.sin(2 * np.pi * x_true)

degree = 8
coeffs = np.random.uniform(-0.5, 0.5, degree)
```

I dati sono stati generati utilizzando le funzionalità offerte dalla libreria `numpy`, facendo uso esclusivo di array. Tale scelta permette di eseguire le operazioni in forma vettorializzata, con un conseguente miglioramento dell’efficienza computazionale rispetto a un'implementazione basata su iterazioni esplicite. Inoltre, questa modalità di gestione dei dati garantisce una buona scalabilità del codice, che rimane efficiente anche aumentando il numero di campioni `n_samples`.

In tutte le esecuzioni presentate in questa relazione sono stati utilizzati 200 punti generati secondo la procedura illustrata sopra: i valori di `x` sono estratti casualmente da una distribuzione uniforme nell’intervallo $[0, 1]$, mentre i corrispondenti valori di `y` sono ottenuti a partire dalla funzione seno, con l’aggiunta di un termine di rumore. Quando non specificato, il valore del parametro `eta` (ossia il _learning rate_) è stato fissato a $0.5$.

== Modello di regressione polinomiale

```py
def polynomial_model(coeffs: np.ndarray, x: np.ndarray):
    powers = x[:, None] ** np.arange(len(coeffs))
    return powers @ coeffs
```

La funzione riportata sopra implementa il modello di regressione polinomiale. Essa riceve in ingresso un array di coefficienti e un array di valori `x`, e restituisce l’array dei corrispondenti valori predetti `y`, associati uno a uno agli elementi di input.

Sebbene l’implementazione possa apparire inizialmente poco intuitiva, la sua struttura è pensata per sfruttare la vettorializzazione offerta da `numpy`, così da eseguire il calcolo in modo efficiente su tutti i punti simultaneamente.

In particolare, l’istruzione `x[:, None]` trasforma il vettore `x` in un array colonna, mentre `np.arange(len(coeffs))` genera il vettore degli esponenti del polinomio, cioè `[0, 1, ..., g]`, dove `g` rappresenta il grado del polinomio. L’operatore `**` viene quindi utilizzato per elevare ciascun valore di `x` a tutte le potenze richieste, costruendo così la matrice delle potenze:

$
"powers" = mat(
  1, x_1, x_1^2, ..., x_1^g;
  1, x_2, x_2^2, ..., x_2^g;
  dots.v, dots.v, dots.v, dots.down, dots.v;
  1, x_N, x_N^2, ..., x_N^g
)
$

A questo punto, il prodotto matrice-vettore tra `powers` e `coeffs` consente di calcolare simultaneamente il valore del polinomio in tutti i punti considerati.

Questo approccio richiede un maggiore utilizzo di memoria rispetto a una versione iterativa, poiché costruisce esplicitamente la matrice delle potenze, ma risulta generalmente più efficiente in termini di tempo di esecuzione grazie all’ottimizzazione delle operazioni vettoriali.

== Discesa del gradiente

La formula analitica per il calcolo del gradiente con funzione costo MSE e modello polinomiale è:

$ gradient_bold("W") J(bold("W")) = -1/N
vec(
  sum_(i=1)^N ( y_i - hat(y)_i ) x_i^0 ,
  sum_(i=1)^N ( y_i - hat(y)_i ) x_i^1 ,
  sum_(i=1)^N ( y_i - hat(y)_i ) x_i^2 ,
  dots.v ,
  sum_(i=1)^N ( y_i - hat(y)_i ) x_i^d
) $


L'implementazione in python è:

```py
def gradient_cost_function(
    coeffs: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    predictions = polynomial_model(coeffs, x)
    errors = predictions - y
    
    X_poly = x[:, None] ** np.arange(len(coeffs))
    gradient = errors @ X_poly / len(x) 
    return gradient
```

Come prima si utilizza il calcolo matriciale come miglior implementazione del codice (nella prima versione del codice era presente un ciclo for che iterava sui gradi, con la nuova implementazione si è osservato un miglioramente del 2x, raddoppiando il numero di iterazioni per secondo)

L'effettiva discesa del gradiente è poi implementata all'interno del seguente ciclo for:
```py
with tqdm(range(40000), desc="Ottimizzazione", unit="iter") as t:
        for i in t:
            grad = gradient_cost_function(coeffs, x, y)
            cost = cost_function(coeffs, x, y)
            t.set_postfix({"Costo": f"{cost:.6e}"})
            coeffs -= eta * grad
```

La prima riga di questo snippet serve semplicemente a creare una grafica confortevole durante il training che dia una barra di caricamento grafica e mostri se il valore della funzione costo (che spiegata in seguito) diminuisce effettivamente

== Discesa del gradiente

// In questa parte aggiungere qualche grafichino per monitorare il camio di eta che cosa porta

La formula analitica per il calcolo del gradiente, nel caso di funzione costo MSE e modello di regressione polinomiale, è la seguente:

$
gradient_bold("W") J(bold("W")) = -1/N
vec(
  sum_(i=1)^N ( y_i - hat(y)_i ) x_i^0 ,
  sum_(i=1)^N ( y_i - hat(y)_i ) x_i^1 ,
  sum_(i=1)^N ( y_i - hat(y)_i ) x_i^2 ,
  dots.v ,
  sum_(i=1)^N ( y_i - hat(y)_i ) x_i^d
)
$

L’implementazione in Python è la seguente:


```py
def gradient_cost_function(
    coeffs: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    predictions = polynomial_model(coeffs, x)
    errors = predictions - y
    
    X_poly = x[:, None] ** np.arange(len(coeffs))
    gradient = errors @ X_poly / len(x) 
    return gradient
```

Anche in questo caso si è scelto di utilizzare un’implementazione vettorializzata basata sul calcolo matriciale, in modo analogo a quanto fatto per il modello polinomiale. Questa soluzione risulta più efficiente rispetto a un approccio iterativo esplicito. Nella prima versione del codice, infatti, il gradiente veniva calcolato mediante un ciclo for sui vari gradi del polinomio; la successiva riformulazione matriciale ha permesso di ottenere un miglioramento prestazionale di circa un fattore 2, raddoppiando il numero di iterazioni eseguibili al secondo.

L’aggiornamento iterativo dei coefficienti nel main del programma tramite discesa del gradiente è poi implementato nel seguente ciclo:

```py
with tqdm(range(40000), desc="Ottimizzazione", unit="iter") as t:
        for i in t:
            grad = gradient_cost_function(coeffs, x, y)
            cost = cost_function(coeffs, x, y)
            t.set_postfix({"Costo": f"{cost:.6e}"})
            coeffs -= eta * grad
```

La prima riga di questo frammento di codice ha una funzione esclusivamente pratica: consente di visualizzare una barra di avanzamento durante il processo di ottimizzazione, rendendo più agevole il monitoraggio dell’esecuzione. Inoltre, permette di osservare in tempo reale l’andamento del valore della funzione costo, verificando che esso diminuisca effettivamente nel corso del training. È stata utillizzata a tale scopo la libreria `tqdm`

#pagebreak()

== Mini batch code <sez:minib>

```py
def get_batches(x, y, batch_size):
    indices = np.arange(len(x))
    np.random.shuffle(indices) 

    for i in range(0, len(x), batch_size):
        batch_idx = indices[i : i + batch_size]
        yield x[batch_idx], y[batch_idx]

        
epochs = []
times = []
conv_cost = 6.7e-3
coeffs_init = np.random.uniform(-0.5, 0.5, degree)

for batch_size in range(1, 101):
    start_time = time.perf_counter()

    epoch = 0
    coeffs = coeffs_init.copy()
    cost = cost_function(coeffs, x, y)
    
    while cost > conv_cost:
        epoch += 1

        for x_batch, y_batch in get_batches(x, y, batch_size):
            grad = gradient_cost_function(coeffs, x_batch, y_batch)
            coeffs -= eta * grad

        cost = cost_function(coeffs, x, y)

    epochs.append(epoch)
    end_time = time.perf_counter()
    times.append(end_time - start_time)
```

L’implementazione della variante Mini-batch Stochastic Gradient Descent è riportata nel codice precedente. L’elemento centrale è la funzione ```py get_batches(x, y, batch_size)```, che ha il compito di suddividere il dataset in sottoinsiemi casuali di dimensione controllata.

Il funzionamento può essere descritto nei seguenti passaggi:

- viene creato un array contenente gli indici del dataset;
- tali indici vengono mescolati casualmente a ogni epoca;
- l’istruzione ```py range(0, len(x), batch_size)``` suddivide implicitamente gli indici in blocchi consecutivi della dimensione desiderata;
- tramite selezione indicizzata vengono estratti i corrispondenti sottoinsiemi di x e y;
- infine, la keyword ```py yield``` consente di restituire un batch alla volta senza terminare l’esecuzione della funzione, preservandone lo stato interno tra una chiamata e la successiva.

Quest’ultimo aspetto è particolarmente utile in questo contesto, poiché permette di generare i mini-batch in modo progressivo durante l’iterazione, senza doverli memorizzare tutti simultaneamente in una struttura dati separata.

All’interno del ciclo principale, l’algoritmo segue una logica del tutto analoga a quella della discesa del gradiente standard. Sono inoltre presenti alcune istruzioni aggiuntive per misurare il tempo necessario al raggiungimento della convergenza e confrontare così l’efficienza del metodo al variare di batch_size.





Quello qui sopra è l'implementazione dell'algoritmo con mini batch, tutto gira in torno alla funzione ```py get_batches(x, y, batch_size)``` essa infatti restituisce i batch ma mano che viene chiamata, in ordine:

- Crea una array di indici
- Li mescola
- La funzione ```py range(0, len(x), batch_size)``` divede i 200 indici nei gruppi desiderati
- Classica selezione sugli indici
- Questa è la parte interessante, si utilizza la keyword ```py yield``` che in pratica permette di far restituire un valore alla funzione _congelandone_ lo stato, in modo che quando verra ri-interrogata (in questo caso dal for), la nostra funzione procederà a restituire un altro mini-banch

All'interno del ciclo l'implementazione è simile alla discesa del gradiente classica con qualche riga per fare il benchmarking


= Risultati

== Monitoraggio della convergenza

=== Funzione costo

La funzione costo è implementata come segue:

```py
def cost_function(coeffs: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    predictions = polynomial_model(coeffs, x)
    errors = predictions - y
    return np.mean(errors**2) / 2
```

Si calcola il vettore `predictions` applicando il modello polinomiale ai dati di input x. Successivamente, il vettore `errors` contiene le differenze tra le predizioni e i valori osservati y. Infine, la funzione restituisce la media dei quadrati degli errori divisa per 2, in accordo con la definizione della funzione di costo richiesta.

#figure(
  image("FunzioneCosto.svg", width: 100%),
  caption: [Andamento della funzione costo in funzione del numero di iterazioni, fino a un massimo di $20 space 000$. Le curve, rappresentate con colori diversi, corrispondono a differenti valori del _learning rate_. Per rendere il confronto significativo, i coefficienti del modello sono stati inizializzati agli stessi valori casuali per ogni valore di `eta`, così da far partire l’ottimizzazione dalla medesima condizione iniziale.]
) <fig:fc>

Come si può osservare in @fig:fc, valori più elevati del _learning rate_ portano generalmente a una diminuzione più rapida della funzione costo, consentendo di raggiungere valori più bassi in un numero inferiore di iterazioni. Questo comportamento è coerente con il fatto che, aumentando la dimensione del passo di aggiornamento, l’algoritmo percorre più velocemente la direzione di discesa.

Tuttavia, tale vantaggio è accompagnato da una minore stabilità numerica. Se il _learning rate_ assume valori eccessivamente elevati, gli aggiornamenti dei coefficienti possono diventare troppo bruschi, compromettendo la convergenza dell’algoritmo.

In una seconda _run_ del programma, riportata in @fig:fc2, è stato possibile evidenziare questo aspetto. In questo caso si osserva che alcuni valori di `eta`, pur risultando efficaci in altre condizioni iniziali, non garantiscono più la convergenza del metodo. In particolare, per `eta = 1.21` il processo di ottimizzazione risulta instabile e non converge correttamente.

Questo risultato mostra come la scelta del _learning rate_ non dipenda esclusivamente dalla forma della funzione costo, ma anche dalla posizione iniziale nel spazio dei parametri. In generale, valori troppo elevati di `eta` possono compromettere la robustezza del processo di ottimizzazione, rendendo il comportamento dell’algoritmo fortemente sensibile alle condizioni iniziali.

#figure(
  image("FunzioneCosto2.svg", width: 100%),
  caption: [Secondo confronto dell’andamento della funzione costo al variare del learning rate, si evidenzia come `eta` alti siano più instabili e possano non convergere al minimo della funzione costo.]
) <fig:fc2>
=== Fit della curva

In tutti i fit riportati in questa sezione è stato utilizzato un valore di `eta = 0.5`. Dalle prove preliminari è emerso che questo valore rappresenta un buon compromesso tra rapidità di convergenza e stabilità numerica, consentendo all’algoritmo di raggiungere risultati soddisfacenti in tempi contenuti senza introdurre, nella maggior parte dei casi, comportamenti instabili.

#figure(
  image("grado7.svg", width: 100%),
  caption: [Risultato del fit polinomiale utilizzando un polinomio di grado 7.]
) <fig:curvaGrado7>

#figure(
  image("grado9.svg", width: 100%),
  caption: [Risultato del fit polinomiale utilizzando un polinomio di grado 9.]
) <fig:curvaGrado9>

#figure(
  image("grado14.svg", width: 100%),
  caption: [Risultato del fit polinomiale utilizzando un polinomio di grado 14.]
) <fig:curvaGrado14>

Confrontando i risultati ottenuti per diversi gradi del polinomio, è possibile osservare come la complessità del modello influenzi in modo significativo la qualità dell’approssimazione.

Nel caso mostrato in @fig:curvaGrado7, il fit risulta già qualitativamente buono: la curva approssima in maniera ragionevole l’andamento generale dei dati.

In @fig:curvaGrado9 si osserva un lieve miglioramento della qualità del fit. La curva segue in modo più fedele l’andamento dei punti sperimentali e della funzione sottostante, rappresentando un buon compromesso tra flessibilità del modello e capacità di generalizzazione. Tra i casi analizzati, questo è risultato essere uno dei più efficaci, riuscendo a migliorare l’adattamento senza introdurre in maniera marcata fenomeni di _over-fitting_.

Aumentando la complessità del modello, quindi il grado del polinomio, come mostrato in @fig:curvaGrado14, si acquisisce una maggiore capacità di adattarsi ai dati. Tuttavia, iniziano a comparire i primi segnali di un comportamento eccessivamente aderente ai punti osservati. In particolare, nella parte finale della curva essa piega verso il basso avvicinandosi ad una zona lievemente più densa di punti, discostandosi dal vaore del seno originale, si tratta di un primo accenno di _over-fitting_. Per una discussione più approfondita di questo fenomeno si rimanda alla @sez:overfit.

== Over-fitting <sez:overfit>

Per mostrare in modo più evidente il fenomeno dell’_over-fitting_, è stato eseguito un ulteriore esperimento impostando il grado del polinomio a 300. Dopo circa 5 milioni di iterazioni, il risultato ottenuto è riportato in @fig:overfit.

#figure(
  image("overfit2.svg", width: 100%),
  caption: [Esempio di _over-fitting_ ottenuto utilizzando un polinomio di grado 300. Il valore finale della funzione costo è pari a $6.57 times 10^(-3)$.]
) <fig:overfit>

In questo caso il comportamento del modello evidenzia chiaramente i limiti di un’eccessiva complessità. Si osserva come nella regione finale il polinomio tenda a deformarsi sensibilmente pur di passare molto vicino agli ultimi punti campionati.

Questo comportamento permette effettivamente di ridurre ulteriormente il valore della funzione costo, ma a scapito della capacità del modello di rappresentare correttamente l’andamento generale della funzione generatrice.

== Variante Mini-batch Stochastic Gradient Descent

Per confrontare l’effetto della dimensione del batch sul comportamento dell’ottimizzazione, è stata implementata anche una variante di tipo Mini-batch Stochastic Gradient Descent. In questa configurazione, il gradiente non viene calcolato sull’intero dataset a ogni aggiornamento, ma soltanto su sottoinsiemi casuali di dati di cardinalità `batch_size`.

Questo approccio modifica la dinamica della convergenza: batch più piccoli introducono una maggiore componente stocastica negli aggiornamenti, mentre batch più grandi rendono il processo più regolare.

#figure(
  image("batcsize tempo6.7e-3.svg", width: 100%),
  caption: [Tempo necessario per raggiungere la convergenza in funzione della dimensione del batch, utilizzando un polinomio di grado 9. Sull’asse delle x è riportato il valore di `batch_size`, mentre sull’asse delle y è riportato il tempo di esecuzione in secondi. La convergenza è stata definita come il raggiungimento di un valore della funzione costo inferiore a $6.7 times 10^(-3)$. Il tempo minimo osservato è pari a $0.41$ s, corrispondente a `batch_size = 1`, mentre il tempo massimo è pari a $1.83$ s, corrispondente a `batch_size = 100`. I tempi riportati devono essere interpretati in modo qualitativo, poiché sono stati ottenuti tramite una semplice misurazione interna in Python e non mediante una procedura di benchmarking rigorosa.]
) <fig:batchtempo>

#v(2.5em)
#figure(
  image("batcsize epoche6.7e-3.svg", width: 100%),
  caption: [Numero di epoche necessarie per raggiungere la convergenza in funzione della dimensione del batch, utilizzando un polinomio di grado 9. Sull’asse delle x è riportato il valore di `batch_size`, mentre sull’asse delle y è riportato il numero di epoche richieste per ottenere un valore della funzione costo inferiore a $6.7 times 10^(-3)$. Il numero minimo di epoche osservato è pari a 171, corrispondente a `batch_size = 1`, mentre il massimo è pari a 21431, corrispondente a `batch_size = 100`.]
) <fig:batchtepoche>

L’analisi dei tempi riportati in @fig:batchtempo mostra un comportamento interessante al variare della dimensione del batch. Al netto delle piccole fluttuazioni osservabili tra barre vicine — attribuibili sia all’imprecisione intrinseca della misurazione dei tempi sia alla componente casuale introdotta dalla generazione dei batch — si osserva un andamento complessivamente crescente.

Il metodo che converge più rapidamente risulta essere quello con `batch_size = 1`, cioè il caso in cui ogni aggiornamento viene effettuato utilizzando un singolo punto del dataset. Questo risultato appare sospetto, poiché ci si aspetterebbe che un gradiente così rumoroso comprometta la stabilità e, più in generale, la convergenza del processo di ottimizzazione.

Le prestazioni peggiori si osservano invece per `batch_size = 100`, che corrisponde di fatto a una suddivisione del dataset in soli due batch. In questo caso, il tempo di convergenza raggiunge circa $1.83$ s, contro i circa $0.41$ s osservati per `batch_size = 1`.

In @fig:batchtepoche si osserva un incremento del numero di epoche all’aumentare di `batch_size`, con un andamento a gradoni. Tale comportamento è dovuto al fatto che batch più grandi riducono il numero di aggiornamenti effettuati per epoca; di conseguenza, per raggiungere la convergenza sono necessarie più epoche. La struttura a gradoni è anch’essa attesa, poiché il numero di batch per epoca varia in modo discreto: ogni volta che l’aumento di `batch_size` comporta una riduzione di un’unità nel numero totale di batch necessari a coprire i 200 punti del dataset, si osserva un salto nel numero di epoche richieste.

#figure(
  image("costepoch.svg"),
  caption: [Andamento della funzione costo in funzione del numero di epoche nel caso `batch_size = 15`, utilizzando un polinomio di grado 9. Il grafico evidenzia il carattere rumoroso della convergenza tipico della discesa del gradiente stocastica su mini-batch.]
) <fig:costepoch>

Dal punto di vista qualitativo, i batch più piccoli tendono a rendere il processo di ottimizzazione più rumoroso, poiché ogni aggiornamento viene calcolato su un sottoinsieme ridotto di dati. Questo comportamento è ben visibile in @fig:costepoch, dove, pur osservandosi una tendenza complessiva alla diminuzione della funzione costo, l’andamento risulta irregolare e soggetto a oscillazioni anche marcate.

Questo grafico permette anche di reinterpretare i risultati precedenti. Per come è stato implementato il criterio di arresto utilizzato per generare la @fig:batchtempo — cioè interrompendo l’ottimizzazione non appena la funzione costo scendeva al di sotto di un valore target prefissato — il caso `batch_size = 1` non corrisponde a una convergenza stabile in senso stretto. Più precisamente, il processo può raggiungere in modo casuale un valore sufficientemente basso della funzione costo, senza che ciò implichi un effettivo assestamento della dinamica di ottimizzazione. In queste condizioni, l’algoritmo può infatti arrestarsi in corrispondenza di minimi locali oppure su valori della funzione costo semplicemente bassi, ma non realmente rappresentativi di una soluzione ottimale. In un problema di fitting questo aspetto può talvolta essere riconosciuto visivamente osservando la curva ottenuta, mentre in contesti di machine learning più generali tale valutazione risulta meno immediata.