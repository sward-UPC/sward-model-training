"""Familia de modelos tabulares para la comparación de Knowledge Tracing de SWARD.

Complementa `compare_models.py` con los clasificadores clásicos de scikit-learn
—los mismos que una herramienta de AutoML como PyCaret pondría en su tabla— para
responder una pregunta que un jurado puede hacer con razón: *¿un clasificador
corriente sobre variables agregadas iguala al modelo secuencial?*

**Por qué hace falta aplanar.** SAKT y DKT leen la secuencia completa. Un
clasificador tabular recibe una fila por predicción, así que la historia del
estudiante hay que resumirla en variables. Eso es exactamente lo que se pierde al
comparar por este camino, y por eso estos modelos son un **baseline**, no un
reemplazo: ninguno produce mapas de atención, que es lo que sostiene la
explicación.

**Sin fuga de información.** Es el riesgo real de este enfoque y se controla en
dos sitios:

1. Las variables de la historia del estudiante se calculan **solo con el prefijo**
   de su propia secuencia (posiciones 0..t) para predecir la posición t+1. Nunca
   se mira hacia adelante.
2. Las dificultades por concepto se calculan **solo con las secuencias de train**
   del fold. Un concepto que no aparece en train cae al promedio global de train.

Se respeta el mismo par (entrada, objetivo) que el resto del script: en el paso t
se conoce (concepts[t], responses[t]) y se predice responses[t+1] del concepto
concepts[t+1].
"""

from __future__ import annotations

import math

# Nombre corto -> etiqueta para la tabla. El orden es el de la tabla.
MODELOS = {
    "tab_dummy": "Dummy (clase mayoritaria)",
    "tab_logistic": "Regresión logística",
    "tab_ridge": "Ridge",
    "tab_sgd": "SGD (logística)",
    "tab_lda": "Análisis discriminante lineal",
    "tab_qda": "Análisis discriminante cuadrático",
    "tab_gaussian_nb": "Naive Bayes gaussiano",
    "tab_bernoulli_nb": "Naive Bayes de Bernoulli",
    "tab_knn": "K vecinos más cercanos",
    "tab_tree": "Árbol de decisión",
    "tab_extra_tree": "Árbol extra",
    "tab_random_forest": "Random Forest",
    "tab_extra_trees": "Extra Trees",
    "tab_gradient_boosting": "Gradient Boosting",
    "tab_hist_gradient_boosting": "Hist. Gradient Boosting",
    "tab_adaboost": "AdaBoost",
    "tab_bagging": "Bagging",
    "tab_svc": "SVM (RBF)",
    "tab_mlp": "Perceptrón multicapa",
}

NOMBRES_VARIABLES = [
    "n_previas",
    "acc_previa",
    "acc_ultimas3",
    "ultima_respuesta",
    "racha",
    "n_previas_concepto",
    "acc_previa_concepto",
    "pasos_desde_concepto",
    "visto_antes_concepto",
    "dificultad_concepto",
    "soporte_concepto",
    "dificultad_concepto_actual",
    "posicion_relativa",
    "mismo_concepto",
]


# ──────────────────────────────────────────────────────────────────────────────
# Construcción de variables
# ──────────────────────────────────────────────────────────────────────────────
def _estadisticos_de_train(train: list[dict], n_skills: int) -> tuple[list[float], list[int], float]:
    """Dificultad media y soporte de cada concepto, calculados solo sobre train."""
    suma = [0.0] * n_skills
    cuenta = [0] * n_skills
    todas = 0
    aciertos = 0
    for s in train:
        for c, r in zip(s["concepts"], s["responses"]):
            todas += 1
            aciertos += r
            if 0 <= c < n_skills:
                suma[c] += r
                cuenta[c] += 1
    global_p = (aciertos / todas) if todas else 0.5
    dificultad = [(suma[i] / cuenta[i]) if cuenta[i] else global_p for i in range(n_skills)]
    return dificultad, cuenta, global_p


def _filas_de_secuencia(
    seq: dict,
    dificultad: list[float],
    soporte: list[int],
    global_p: float,
    n_skills: int,
) -> tuple[list[list[float]], list[int]]:
    """Convierte una secuencia en filas (X) y objetivos (y), de forma causal."""
    c, r = seq["concepts"], seq["responses"]
    L = len(c)
    X: list[list[float]] = []
    y: list[int] = []

    acumulado = 0
    for t in range(L - 1):
        acumulado += r[t]
        n_prev = t + 1
        cq = c[t + 1]

        # Historia del propio concepto objetivo, solo hasta t.
        n_cq = 0
        ac_cq = 0
        ultimo_cq = -1
        for i in range(t + 1):
            if c[i] == cq:
                n_cq += 1
                ac_cq += r[i]
                ultimo_cq = i

        # Racha de respuestas iguales terminando en t, con signo.
        racha = 1
        while racha <= t and r[t - racha] == r[t]:
            racha += 1
        racha_firmada = racha if r[t] == 1 else -racha

        ventana = r[max(0, t - 2) : t + 1]

        dif_cq = dificultad[cq] if 0 <= cq < n_skills else global_p
        sop_cq = soporte[cq] if 0 <= cq < n_skills else 0
        dif_actual = dificultad[c[t]] if 0 <= c[t] < n_skills else global_p

        X.append([
            float(n_prev),
            acumulado / n_prev,
            sum(ventana) / len(ventana),
            float(r[t]),
            float(racha_firmada),
            float(n_cq),
            (ac_cq / n_cq) if n_cq else global_p,
            float(t - ultimo_cq) if ultimo_cq >= 0 else 0.0,
            1.0 if n_cq else 0.0,
            dif_cq,
            math.log1p(sop_cq),
            dif_actual,
            n_prev / L,
            1.0 if c[t] == cq else 0.0,
        ])
        y.append(r[t + 1])

    return X, y


def construir_matrices(train: list[dict], test: list[dict], n_skills: int):
    """Devuelve (X_train, y_train, X_test, y_test) sin fuga de información."""
    dificultad, soporte, global_p = _estadisticos_de_train(train, n_skills)

    Xtr: list[list[float]] = []
    ytr: list[int] = []
    for s in train:
        x, y = _filas_de_secuencia(s, dificultad, soporte, global_p, n_skills)
        Xtr.extend(x)
        ytr.extend(y)

    Xte: list[list[float]] = []
    yte: list[int] = []
    for s in test:
        x, y = _filas_de_secuencia(s, dificultad, soporte, global_p, n_skills)
        Xte.extend(x)
        yte.extend(y)

    return Xtr, ytr, Xte, yte


# ──────────────────────────────────────────────────────────────────────────────
# Constructores
# ──────────────────────────────────────────────────────────────────────────────
def _constructor(nombre: str, semilla: int):
    """Devuelve (estimador, necesita_escalado). Importa sklearn de forma perezosa."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        AdaBoostClassifier,
        BaggingClassifier,
    )
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    tabla = {
        "tab_dummy": (DummyClassifier(strategy="prior"), False),
        "tab_logistic": (LogisticRegression(max_iter=2000, random_state=semilla), True),
        "tab_ridge": (RidgeClassifier(random_state=semilla), True),
        "tab_sgd": (
            SGDClassifier(loss="log_loss", max_iter=3000, tol=1e-4, random_state=semilla),
            True,
        ),
        "tab_lda": (LinearDiscriminantAnalysis(), True),
        "tab_qda": (QuadraticDiscriminantAnalysis(reg_param=0.1), True),
        "tab_gaussian_nb": (GaussianNB(), True),
        "tab_bernoulli_nb": (BernoulliNB(), True),
        "tab_knn": (KNeighborsClassifier(n_neighbors=15), True),
        "tab_tree": (DecisionTreeClassifier(max_depth=6, random_state=semilla), False),
        "tab_extra_tree": (ExtraTreeClassifier(max_depth=6, random_state=semilla), False),
        "tab_random_forest": (
            RandomForestClassifier(n_estimators=300, random_state=semilla, n_jobs=1),
            False,
        ),
        "tab_extra_trees": (
            ExtraTreesClassifier(n_estimators=300, random_state=semilla, n_jobs=1),
            False,
        ),
        "tab_gradient_boosting": (GradientBoostingClassifier(random_state=semilla), False),
        "tab_hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=semilla),
            False,
        ),
        "tab_adaboost": (AdaBoostClassifier(random_state=semilla), False),
        "tab_bagging": (BaggingClassifier(random_state=semilla, n_jobs=1), False),
        "tab_svc": (SVC(probability=True, random_state=semilla), True),
        "tab_mlp": (
            MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=semilla),
            True,
        ),
    }
    if nombre not in tabla:
        raise ValueError("Modelo tabular desconocido: %s" % nombre)
    return tabla[nombre]


def _probabilidades(modelo, X):
    """Probabilidad de la clase 1, sea cual sea la interfaz del estimador."""
    if hasattr(modelo, "predict_proba"):
        p = modelo.predict_proba(X)
        clases = list(modelo.classes_)
        if 1 in clases:
            return [float(fila[clases.index(1)]) for fila in p]
        return [0.0 for _ in p]
    # RidgeClassifier y similares: se pasa la función de decisión por una logística.
    if hasattr(modelo, "decision_function"):
        d = modelo.decision_function(X)
        return [1.0 / (1.0 + math.exp(-float(v))) for v in d]
    return [float(v) for v in modelo.predict(X)]


def entrenar_y_predecir(
    nombre: str, train: list[dict], test: list[dict], n_skills: int, cfg: dict
) -> tuple[list[int], list[float]]:
    """Entrena un modelo tabular en el fold y devuelve (y_true, y_score)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr, Xte, yte = construir_matrices(train, test, n_skills)
    if not Xtr or not Xte:
        return [], []

    estimador, escalar = _constructor(nombre, cfg["seed"])
    modelo = make_pipeline(StandardScaler(), estimador) if escalar else estimador

    # Un fold con una sola clase en train no se puede entrenar: se devuelve la
    # tasa base, que es lo que haría cualquier clasificador razonable.
    if len(set(ytr)) < 2:
        p = sum(ytr) / len(ytr)
        return yte, [p] * len(yte)

    modelo.fit(Xtr, ytr)
    return yte, _probabilidades(modelo, Xte)
