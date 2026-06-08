AC Python Demo — Machine Learning Utilities (CPU/GPU)

Un insieme di wrapper Python progettati per semplificare attività di Machine Learning su dataset tabellari, con supporto automatico a backend Scikit‑Learn (CPU) e RAPIDS cuML (GPU).
Include esempi completi, report generati automaticamente e notebook dimostrativi.


*** CARATTERISTICHE PRINCIPALI ***

UniversalClassifier — wrapper unificato per classificazione
Supporta Decision Tree, Random Forest, SVM, Dummy, con:

    - selezione automatica del backend (CPU/GPU)
    - gridsearch integrato
    - metriche, confusion matrix, ROC, performance curves
    - esportazione automatica dei risultati in PDF

ClusterMaker — wrapper per clustering
Supporta KMeans, DBSCAN, Agglomerative, con:

    - SSE curve
    - scelta automatica degli iperparametri
    - visualizzazioni 2D con centriide
    - pipeline di scaling e riduzione dimensionale

Supporto GPU trasparente  
Se una GPU NVIDIA è disponibile, i wrapper passano automaticamente a cuML.

Report — generatore di report PDF da DataFrame
Ideale per pipeline di analisi automatizzate.


*** NOTEBOOK DIMOSTRATIVI E DOCUMENTI ***

    - demo_classification.ipynb: dimostra l’uso del classificatore universale, la ricerca degli iperparametri e la generazione di report.
    - demo_clustering.ipynb: esempio completo di clustering su dati reali (tweet della Presidenza USA).
    - PDF già generati nella cartella reports/pdf


*** STRUTTURA DEL REPOSITORY ***
Code

AC_Python_Demo/
│
├── classification/           # Wrapper UniversalClassifier
├── clustering/               # Wrapper ClusterMaker
├── tabular/                  # Report generator
│
├── examples/
│   ├── demo_classification.ipynb
│   ├── demo_clustering.ipynb
│   └── pdf/                  # PDF generati dai notebook
│
├── data/                     # Dataset di esempio
│
├── img/                      # Immagini per README e report
│
└── README.md


*** ESEMPI RAPIDI ***

Classificazione
    
    from ac.ml.classification import UniversalClassifier
    import pandas as pd
    
    df = pd.read_csv("billionaires.csv")
    
    uc = UniversalClassifier(
        method="randomforest",
        data=df,
        target="Mapped_Country",
        subset=["Wealth", "Mapped_Region", "Mapped_Sector"]
    )
    
    uc.split("30%")
    uc.explore(param={"max_depth": [4, 8, 16]}, cv=5)
    uc.fit(cv=5)
    
    print("Prediction:", uc.predict([80, 1.0, 3.0]))
    uc.draw("confusion", set="test", show=True)

> Clustering

    from ac.ml.clustering import ClusterMaker
    import pandas as pd
    
    df = pd.read_csv("small_presidential_tweets.csv")
    
    cm = ClusterMaker(
        method="kmeans",
        data=df,
        subset=["admiration", "gratitude"],
        scaling="minmax"
    )
    
    cm.explore()
    cm.fit()
    cm.draw("clustering", x="admiration", y="gratitude", centroids=True, show=True)


Report automatici
Il modulo Report consente di generare PDF direttamente da DataFrame:
    
    from ac.tabular import Report
    
    report = Report()
    report.set(source=df)
    report.headline("My Report")
    report.dump("output", extension=".pdf")


*** REQUISITI ***

- Python 3.10+
- Scikit‑Learn
- RAPIDS cuML (opzionale, per GPU)
- Matplotlib, Seaborn, Pandas, NumPy


*** AUTORE ***

Alessio Cioli  
Docente, sviluppatore Python e studente di Data Science & AI presso Università di Pisa

Per domande o collaborazioni:
alessio.cioli@yahoo.it
