"""
Implémentation des Nuées dynamiques FROM SCRATCH.

Les Nuées dynamiques sont ici considérées comme un cadre général
permettant plusieurs types de représentations des classes.

Représentations disponibles :
1. Point représentatif
2. Ensemble de points représentatifs
3. Axes / composantes factorielles
4. Distribution de probabilités
5. Structure représentative

Aucune fonction KMeans de Scikit-learn n'est utilisée.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. GÉNÉRATION DES DONNÉES
# ============================================================

def generer_donnees():
    """
    Génère un jeu de données artificiel de 90 observations
    réparties autour de trois groupes.
    """

    rng = np.random.default_rng(42)

    groupe1 = rng.normal(
        loc=[2, 2],
        scale=[0.55, 0.55],
        size=(30, 2)
    )

    groupe2 = rng.normal(
        loc=[7, 3],
        scale=[0.60, 0.50],
        size=(30, 2)
    )

    groupe3 = rng.normal(
        loc=[4, 8],
        scale=[0.50, 0.60],
        size=(30, 2)
    )

    return np.vstack([
        groupe1,
        groupe2,
        groupe3
    ])


# ============================================================
# 2. DISTANCE EUCLIDIENNE
# ============================================================

def distance_euclidienne(a, b):
    """
    Calcule la distance euclidienne entre deux points.
    """

    return np.sqrt(
        np.sum((a - b) ** 2)
    )


# ============================================================
# 3. POINT REPRÉSENTATIF
# ============================================================

def initialiser_points(X, k, seed=42):
    """
    Initialise un point représentatif par classe.
    """

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(X),
        size=k,
        replace=False
    )

    return X[indices].copy()


def distance_point(x, representation):
    """
    Distance entre une observation et un point représentatif.
    """

    return distance_euclidienne(
        x,
        representation
    )


def mettre_a_jour_points(
    X,
    etiquettes,
    representations
):
    """
    Recalcule le centroïde de chaque classe.
    """

    nouveaux = np.zeros_like(
        representations
    )

    for j in range(len(representations)):

        points = X[
            etiquettes == j
        ]

        if len(points) == 0:
            nouveaux[j] = representations[j]
        else:
            nouveaux[j] = np.mean(
                points,
                axis=0
            )

    return nouveaux


# ============================================================
# 4. ENSEMBLE DE POINTS REPRÉSENTATIFS
# ============================================================

def initialiser_ensembles(
    X,
    k,
    nombre_representants,
    seed=42
):
    """
    Initialise plusieurs représentants pour chaque classe.
    """

    rng = np.random.default_rng(seed)

    ensembles = []

    for _ in range(k):

        indices = rng.choice(
            len(X),
            size=nombre_representants,
            replace=False
        )

        ensembles.append(
            X[indices].copy()
        )

    return ensembles


def distance_ensemble(
    x,
    representations
):
    """
    Distance d'une observation à un ensemble de points.
    On retient la distance au représentant le plus proche.
    """

    distances = [
        distance_euclidienne(x, p)
        for p in representations
    ]

    return min(distances)


def affecter_ensemble(
    X,
    representations
):
    """
    Affecte chaque observation à l'ensemble
    de représentants le plus proche.
    """

    etiquettes = np.zeros(
        len(X),
        dtype=int
    )

    for i, x in enumerate(X):

        distances = [
            distance_ensemble(
                x,
                representations[j]
            )
            for j in range(len(representations))
        ]

        etiquettes[i] = np.argmin(
            distances
        )

    return etiquettes


def mettre_a_jour_ensembles(
    X,
    etiquettes,
    representations
):
    """
    Met à jour les représentants de chaque classe.

    Chaque point d'une classe est affecté au représentant
    le plus proche, puis chaque représentant est remplacé
    par la moyenne des points qui lui sont associés.
    """

    nouveaux = []

    for classe in range(
        len(representations)
    ):

        points_classe = X[
            etiquettes == classe
        ]

        anciens = representations[classe]
        nouveaux_classe = anciens.copy()

        if len(points_classe) == 0:
            nouveaux.append(
                nouveaux_classe
            )
            continue

        sous_groupes = [
            []
            for _ in range(
                len(anciens)
            )
        ]

        for point in points_classe:

            distances = [
                distance_euclidienne(
                    point,
                    centre
                )
                for centre in anciens
            ]

            indice = np.argmin(
                distances
            )

            sous_groupes[indice].append(
                point
            )

        for j, groupe in enumerate(
            sous_groupes
        ):

            if len(groupe) > 0:

                nouveaux_classe[j] = np.mean(
                    groupe,
                    axis=0
                )

        nouveaux.append(
            nouveaux_classe
        )

    return nouveaux


# ============================================================
# 5. AXES / COMPOSANTES FACTORIELLES
# ============================================================

def calculer_axes(points, nombre_axes=1):
    """
    Calcule une représentation par centre et axes principaux.

    Les axes sont obtenus à partir des vecteurs propres
    de la matrice de covariance.
    """

    centre = np.mean(
        points,
        axis=0
    )

    if len(points) <= 1:

        axes = np.eye(
            points.shape[1]
        )[:nombre_axes]

        return centre, axes

    donnees_centrees = (
        points - centre
    )

    covariance = np.cov(
        donnees_centrees,
        rowvar=False
    )

    if np.ndim(covariance) == 0:
        covariance = np.array(
            [[float(covariance)]]
        )

    valeurs, vecteurs = np.linalg.eigh(
        covariance
    )

    ordre = np.argsort(
        valeurs
    )[::-1]

    vecteurs = vecteurs[
        :, ordre
    ]

    nombre_axes = min(
        nombre_axes,
        vecteurs.shape[1]
    )

    axes = vecteurs[
        :, :nombre_axes
    ].T

    return centre, axes


def initialiser_axes(
    X,
    k,
    nombre_axes,
    seed=42
):
    """
    Initialise les représentations par axes.
    """

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(X),
        size=k,
        replace=False
    )

    representations = []

    for indice in indices:

        centre = X[indice].copy()

        axes = np.eye(
            X.shape[1]
        )[:nombre_axes]

        representations.append(
            {
                "centre": centre,
                "axes": axes
            }
        )

    return representations


def distance_axes(
    x,
    representation
):
    """
    Distance d'un point à l'espace engendré
    par les axes principaux.
    """

    centre = representation["centre"]
    axes = representation["axes"]

    difference = x - centre

    if len(axes) == 0:
        return np.linalg.norm(
            difference
        )

    projection = np.zeros_like(
        difference
    )

    for axe in axes:

        projection += (
            np.dot(difference, axe)
            * axe
        )

    residu = (
        difference - projection
    )

    return np.linalg.norm(
        residu
    )


def affecter_axes(
    X,
    representations
):
    """
    Affecte chaque observation à la représentation
    par axes la plus proche.
    """

    etiquettes = np.zeros(
        len(X),
        dtype=int
    )

    for i, x in enumerate(X):

        distances = [
            distance_axes(
                x,
                representation
            )
            for representation
            in representations
        ]

        etiquettes[i] = np.argmin(
            distances
        )

    return etiquettes


def mettre_a_jour_axes(
    X,
    etiquettes,
    representations,
    nombre_axes
):
    """
    Recalcule le centre et les axes de chaque classe.
    """

    nouveaux = []

    for classe in range(
        len(representations)
    ):

        points = X[
            etiquettes == classe
        ]

        if len(points) == 0:

            nouveaux.append(
                representations[classe]
            )

        else:

            centre, axes = calculer_axes(
                points,
                nombre_axes
            )

            nouveaux.append(
                {
                    "centre": centre,
                    "axes": axes
                }
            )

    return nouveaux


# ============================================================
# 6. DISTRIBUTION DE PROBABILITÉS
# ============================================================

def creer_distribution(points):
    """
    Représente une classe par une distribution gaussienne
    caractérisée par une moyenne et une matrice de covariance.
    """

    moyenne = np.mean(
        points,
        axis=0
    )

    if len(points) <= 1:

        covariance = np.eye(
            points.shape[1]
        )

    else:

        covariance = np.cov(
            points,
            rowvar=False
        )

        if np.ndim(covariance) == 0:
            covariance = np.array(
                [[float(covariance)]]
            )

    # Régularisation pour éviter une matrice singulière
    covariance = (
        covariance
        + 1e-6 * np.eye(
            covariance.shape[0]
        )
    )

    return {
        "moyenne": moyenne,
        "covariance": covariance
    }


def initialiser_distributions(
    X,
    k,
    seed=42
):
    """
    Initialise les distributions à partir de groupes
    provisoires de données.
    """

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(X),
        size=k,
        replace=False
    )

    distributions = []

    for indice in indices:

        moyenne = X[indice].copy()

        covariance = (
            np.eye(
                X.shape[1]
            )
            * 0.5
        )

        distributions.append(
            {
                "moyenne": moyenne,
                "covariance": covariance
            }
        )

    return distributions


def distance_probabilite(
    x,
    distribution
):
    """
    Calcule une distance basée sur la distance
    de Mahalanobis.
    """

    moyenne = distribution[
        "moyenne"
    ]

    covariance = distribution[
        "covariance"
    ]

    difference = (
        x - moyenne
    )

    inverse = np.linalg.inv(
        covariance
    )

    distance = np.sqrt(
        difference.T
        @ inverse
        @ difference
    )

    return distance


def affecter_distributions(
    X,
    distributions
):
    """
    Affecte chaque observation à la distribution
    la plus proche.
    """

    etiquettes = np.zeros(
        len(X),
        dtype=int
    )

    for i, x in enumerate(X):

        distances = [
            distance_probabilite(
                x,
                distribution
            )
            for distribution
            in distributions
        ]

        etiquettes[i] = np.argmin(
            distances
        )

    return etiquettes


def mettre_a_jour_distributions(
    X,
    etiquettes,
    distributions
):
    """
    Recalcule la moyenne et la covariance
    de chaque classe.
    """

    nouvelles = []

    for classe in range(
        len(distributions)
    ):

        points = X[
            etiquettes == classe
        ]

        if len(points) == 0:

            nouvelles.append(
                distributions[classe]
            )

        else:

            nouvelles.append(
                creer_distribution(
                    points
                )
            )

    return nouvelles


# ============================================================
# 7. STRUCTURE REPRÉSENTATIVE
# ============================================================

def creer_structure(points):
    """
    Représente une classe par une structure
    de type intervalle multidimensionnel.

    Pour chaque variable, on conserve :
    - la valeur minimale ;
    - la valeur maximale.
    """

    minimum = np.min(
        points,
        axis=0
    )

    maximum = np.max(
        points,
        axis=0
    )

    return {
        "minimum": minimum,
        "maximum": maximum
    }


def initialiser_structures(
    X,
    k,
    seed=42
):
    """
    Initialise les structures représentatives.
    """

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(X),
        size=k,
        replace=False
    )

    structures = []

    for indice in indices:

        point = X[indice]

        minimum = point - 0.5
        maximum = point + 0.5

        structures.append(
            {
                "minimum": minimum,
                "maximum": maximum
            }
        )

    return structures


def distance_structure(
    x,
    structure
):
    """
    Distance entre une observation et une structure
    de type intervalle multidimensionnel.

    La distance est nulle si le point est à l'intérieur
    de la structure.
    """

    minimum = structure[
        "minimum"
    ]

    maximum = structure[
        "maximum"
    ]

    distance = np.zeros_like(
        x,
        dtype=float
    )

    for j in range(
        len(x)
    ):

        if x[j] < minimum[j]:

            distance[j] = (
                minimum[j] - x[j]
            )

        elif x[j] > maximum[j]:

            distance[j] = (
                x[j] - maximum[j]
            )

        else:

            distance[j] = 0.0

    return np.linalg.norm(
        distance
    )


def affecter_structures(
    X,
    structures
):
    """
    Affecte chaque observation à la structure
    la plus proche.
    """

    etiquettes = np.zeros(
        len(X),
        dtype=int
    )

    for i, x in enumerate(X):

        distances = [
            distance_structure(
                x,
                structure
            )
            for structure
            in structures
        ]

        etiquettes[i] = np.argmin(
            distances
        )

    return etiquettes


def mettre_a_jour_structures(
    X,
    etiquettes,
    structures
):
    """
    Recalcule les bornes min/max de chaque structure.
    """

    nouvelles = []

    for classe in range(
        len(structures)
    ):

        points = X[
            etiquettes == classe
        ]

        if len(points) == 0:

            nouvelles.append(
                structures[classe]
            )

        else:

            nouvelles.append(
                creer_structure(
                    points
                )
            )

    return nouvelles


# ============================================================
# 8. FONCTIONS GÉNÉRALES
# ============================================================

def initialiser(
    X,
    k,
    type_representation,
    parametre,
    seed=42
):

    if type_representation == 1:

        return initialiser_points(
            X,
            k,
            seed
        )

    elif type_representation == 2:

        return initialiser_ensembles(
            X,
            k,
            parametre,
            seed
        )

    elif type_representation == 3:

        return initialiser_axes(
            X,
            k,
            parametre,
            seed
        )

    elif type_representation == 4:

        return initialiser_distributions(
            X,
            k,
            seed
        )

    elif type_representation == 5:

        return initialiser_structures(
            X,
            k,
            seed
        )

    else:

        raise ValueError(
            "Type de représentation invalide."
        )


def affecter(
    X,
    representations,
    type_representation
):

    if type_representation == 1:

        etiquettes = np.zeros(
            len(X),
            dtype=int
        )

        for i, x in enumerate(X):

            distances = [
                distance_point(
                    x,
                    representation
                )
                for representation
                in representations
            ]

            etiquettes[i] = np.argmin(
                distances
            )

        return etiquettes

    elif type_representation == 2:

        return affecter_ensemble(
            X,
            representations
        )

    elif type_representation == 3:

        return affecter_axes(
            X,
            representations
        )

    elif type_representation == 4:

        return affecter_distributions(
            X,
            representations
        )

    elif type_representation == 5:

        return affecter_structures(
            X,
            representations
        )


def mettre_a_jour(
    X,
    etiquettes,
    representations,
    type_representation,
    parametre
):

    if type_representation == 1:

        return mettre_a_jour_points(
            X,
            etiquettes,
            representations
        )

    elif type_representation == 2:

        return mettre_a_jour_ensembles(
            X,
            etiquettes,
            representations
        )

    elif type_representation == 3:

        return mettre_a_jour_axes(
            X,
            etiquettes,
            representations,
            parametre
        )

    elif type_representation == 4:

        return mettre_a_jour_distributions(
            X,
            etiquettes,
            representations
        )

    elif type_representation == 5:

        return mettre_a_jour_structures(
            X,
            etiquettes,
            representations
        )


def calculer_critere(
    X,
    etiquettes,
    representations,
    type_representation
):
    """
    Calcule le critère global utilisé pour suivre
    l'évolution de l'algorithme.
    """

    critere = 0.0

    for i, x in enumerate(X):

        classe = etiquettes[i]

        if type_representation == 1:

            distance = distance_point(
                x,
                representations[classe]
            )

        elif type_representation == 2:

            distance = distance_ensemble(
                x,
                representations[classe]
            )

        elif type_representation == 3:

            distance = distance_axes(
                x,
                representations[classe]
            )

        elif type_representation == 4:

            distance = distance_probabilite(
                x,
                representations[classe]
            )

        elif type_representation == 5:

            distance = distance_structure(
                x,
                representations[classe]
            )

        critere += distance ** 2

    return critere


def distance_representations(
    anciennes,
    nouvelles,
    type_representation
):
    """
    Mesure le déplacement des représentations.
    """

    distances = []

    if type_representation == 1:

        for ancien, nouveau in zip(
            anciennes,
            nouvelles
        ):

            distances.append(
                distance_euclidienne(
                    ancien,
                    nouveau
                )
            )

    elif type_representation == 2:

        for ancien, nouveau in zip(
            anciennes,
            nouvelles
        ):

            sous_distances = []

            for a, n in zip(
                ancien,
                nouveau
            ):

                sous_distances.append(
                    distance_euclidienne(
                        a,
                        n
                    )
                )

            distances.append(
                max(sous_distances)
            )

    elif type_representation == 3:

        for ancien, nouveau in zip(
            anciennes,
            nouvelles
        ):

            distance_centre = distance_euclidienne(
                ancien["centre"],
                nouveau["centre"]
            )

            distances.append(
                distance_centre
            )

    elif type_representation == 4:

        for ancien, nouveau in zip(
            anciennes,
            nouvelles
        ):

            distance_moyenne = distance_euclidienne(
                ancien["moyenne"],
                nouveau["moyenne"]
            )

            distances.append(
                distance_moyenne
            )

    elif type_representation == 5:

        for ancien, nouveau in zip(
            anciennes,
            nouvelles
        ):

            distance_min = distance_euclidienne(
                ancien["minimum"],
                nouveau["minimum"]
            )

            distance_max = distance_euclidienne(
                ancien["maximum"],
                nouveau["maximum"]
            )

            distances.append(
                max(
                    distance_min,
                    distance_max
                )
            )

    return max(distances)


# ============================================================
# 9. ALGORITHME GÉNÉRAL DES NUÉES DYNAMIQUES
# ============================================================

def nuees_dynamiques(
    X,
    k,
    type_representation,
    parametre=None,
    max_iter=100,
    tolerance=1e-4,
    seed=42
):
    """
    Algorithme général des Nuées dynamiques.

    Les quatre éléments fondamentaux sont :

    1. représentation des classes ;
    2. critère d'affectation ;
    3. mise à jour de la représentation ;
    4. critère de convergence.
    """

    representations = initialiser(
        X,
        k,
        type_representation,
        parametre,
        seed
    )

    historique_critere = []

    for iteration in range(
        1,
        max_iter + 1
    ):

        # -----------------------------------------
        # 1. Affectation
        # -----------------------------------------

        etiquettes = affecter(
            X,
            representations,
            type_representation
        )

        # -----------------------------------------
        # 2. Mise à jour
        # -----------------------------------------

        nouvelles_representations = mettre_a_jour(
            X,
            etiquettes,
            representations,
            type_representation,
            parametre
        )

        # -----------------------------------------
        # 3. Critère d'optimisation
        # -----------------------------------------

        critere = calculer_critere(
            X,
            etiquettes,
            nouvelles_representations,
            type_representation
        )

        historique_critere.append(
            critere
        )

        # -----------------------------------------
        # 4. Convergence
        # -----------------------------------------

        deplacement = distance_representations(
            representations,
            nouvelles_representations,
            type_representation
        )

        print(
            f"Itération {iteration:02d} | "
            f"Critère = {critere:.6f} | "
            f"Déplacement = {deplacement:.8f}"
        )

        representations = (
            nouvelles_representations
        )

        if deplacement < tolerance:

            print(
                f"\nConvergence atteinte "
                f"à l'itération {iteration}."
            )

            break

    return (
        etiquettes,
        representations,
        historique_critere,
        iteration
    )


# ============================================================
# 10. AFFICHAGE DES RÉSULTATS
# ============================================================

def afficher_resultats(
    X,
    etiquettes,
    representations,
    historique,
    type_representation
):
    """
    Affiche les résultats selon la représentation choisie.
    """

    plt.figure(
        figsize=(8, 6)
    )

    nombre_clusters = len(
        representations
    )

    for cluster in range(
        nombre_clusters
    ):

        points = X[
            etiquettes == cluster
        ]

        if len(points) > 0:

            plt.scatter(
                points[:, 0],
                points[:, 1],
                label=f"Classe {cluster + 1}"
            )

    # --------------------------------------------------------
    # Affichage des représentations
    # --------------------------------------------------------

    if type_representation == 1:

        centres = np.array(
            representations
        )

        plt.scatter(
            centres[:, 0],
            centres[:, 1],
            marker="X",
            s=220,
            label="Centres"
        )

    elif type_representation == 2:

        for j, ensemble in enumerate(
            representations
        ):

            ensemble = np.array(
                ensemble
            )

            plt.scatter(
                ensemble[:, 0],
                ensemble[:, 1],
                marker="X",
                s=130,
                label=(
                    f"Représentants "
                    f"classe {j + 1}"
                )
            )

    elif type_representation == 3:

        for j, representation in enumerate(
            representations
        ):

            centre = representation[
                "centre"
            ]

            plt.scatter(
                centre[0],
                centre[1],
                marker="X",
                s=180,
                label=(
                    f"Centre axe {j + 1}"
                )
            )

    elif type_representation == 4:

        moyennes = np.array([
            representation["moyenne"]
            for representation
            in representations
        ])

        plt.scatter(
            moyennes[:, 0],
            moyennes[:, 1],
            marker="X",
            s=220,
            label="Moyennes"
        )

    elif type_representation == 5:

        for j, structure in enumerate(
            representations
        ):

            centre = (
                structure["minimum"]
                + structure["maximum"]
            ) / 2

            plt.scatter(
                centre[0],
                centre[1],
                marker="X",
                s=180,
                label=(
                    f"Structure classe "
                    f"{j + 1}"
                )
            )

    plt.title(
        "Résultat des Nuées dynamiques"
    )

    plt.xlabel(
        "Variable 1"
    )

    plt.ylabel(
        "Variable 2"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        "clusters.png",
        dpi=300
    )

    plt.show()

    # --------------------------------------------------------
    # Critère
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    iterations = range(
        1,
        len(historique) + 1
    )

    plt.plot(
        iterations,
        historique,
        marker="o"
    )

    plt.title(
        "Évolution du critère d'optimisation"
    )

    plt.xlabel(
        "Itération"
    )

    plt.ylabel(
        "Critère"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        "critere.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 11. MENU
# ============================================================

def afficher_menu():
    """
    Affiche le menu de choix de représentation.
    """

    print("\n")
    print("=" * 60)
    print("       APPLICATION DES NUÉES DYNAMIQUES")
    print("=" * 60)

    print("\nChoisissez le type de représentation :")

    print(
        "\n1. Point représentatif"
    )

    print(
        "2. Ensemble de points représentatifs"
    )

    print(
        "3. Axes / composantes factorielles"
    )

    print(
        "4. Distribution de probabilités"
    )

    print(
        "5. Structure représentative"
    )

    print(
        "\n0. Quitter"
    )


# ============================================================
# 12. PROGRAMME PRINCIPAL
# ============================================================

def main():

    X = generer_donnees()

    while True:

        afficher_menu()

        choix = input(
            "\nVotre choix : "
        )

        if choix == "0":

            print(
                "\nFin de l'application."
            )

            break

        if choix not in [
            "1",
            "2",
            "3",
            "4",
            "5"
        ]:

            print(
                "\nChoix invalide."
            )

            continue

        type_representation = int(
            choix
        )

        # ----------------------------------------------------
        # Nombre de classes
        # ----------------------------------------------------

        k = int(
            input(
                "\nNombre de classes K : "
            )
        )

        # ----------------------------------------------------
        # Paramètres spécifiques
        # ----------------------------------------------------

        parametre = None

        if type_representation == 1:

            print(
                "\nReprésentation : "
                "Point représentatif"
            )

            print(
                "Type de point : Centroïde"
            )

            print(
                "Distance : Euclidienne"
            )

        elif type_representation == 2:

            print(
                "\nReprésentation : "
                "Ensemble de points représentatifs"
            )

            parametre = int(
                input(
                    "Nombre de représentants "
                    "par classe : "
                )
            )

            print(
                "Distance : Euclidienne "
                "au représentant le plus proche"
            )

        elif type_representation == 3:

            print(
                "\nReprésentation : "
                "Axes / composantes factorielles"
            )

            parametre = int(
                input(
                    "Nombre d'axes : "
                )
            )

            print(
                "Critère : distance à "
                "l'espace représenté par les axes"
            )

        elif type_representation == 4:

            print(
                "\nReprésentation : "
                "Distribution de probabilités"
            )

            print(
                "Modèle : moyenne + covariance"
            )

            print(
                "Distance : Mahalanobis"
            )

        elif type_representation == 5:

            print(
                "\nReprésentation : "
                "Structure représentative"
            )

            print(
                "Structure : intervalle "
                "multidimensionnel"
            )

            print(
                "Critère : distance à la structure"
            )

        # ----------------------------------------------------
        # Paramètres communs
        # ----------------------------------------------------

        max_iter = int(
            input(
                "\nNombre maximal d'itérations "
                "[100] : "
            )
            or 100
        )

        tolerance = float(
            input(
                "Tolérance [0.0001] : "
            )
            or 0.0001
        )

        # ----------------------------------------------------
        # Exécution
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "EXÉCUTION DES NUÉES DYNAMIQUES"
        )

        print(
            "=" * 60
        )

        (
            etiquettes,
            representations,
            historique,
            nb_iterations
        ) = nuees_dynamiques(
            X,
            k,
            type_representation,
            parametre,
            max_iter,
            tolerance,
            seed=42
        )

        # ----------------------------------------------------
        # Résultats
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "RÉSULTATS FINAUX"
        )

        print(
            "=" * 60
        )

        print(
            f"Nombre de classes : {k}"
        )

        print(
            f"Nombre d'itérations : "
            f"{nb_iterations}"
        )

        print(
            f"Critère final : "
            f"{historique[-1]:.6f}"
        )

        print(
            "\nTaille des classes :"
        )

        for classe in range(k):

            nombre = np.sum(
                etiquettes == classe
            )

            print(
                f"Classe {classe + 1} : "
                f"{nombre} observations"
            )

        afficher_resultats(
            X,
            etiquettes,
            representations,
            historique,
            type_representation
        )

        nouvelle_execution = input(
            "\nVoulez-vous essayer "
            "une autre représentation ? "
            "(o/n) : "
        )

        if nouvelle_execution.lower() != "o":

            print(
                "\nFin de l'application."
            )

            break


# ============================================================
# EXÉCUTION
# ============================================================

if __name__ == "__main__":
    main()