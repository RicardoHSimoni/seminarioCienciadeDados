from flask import Flask, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# =========================
# CARREGAR DADOS
# =========================

ratings = pd.read_csv("dados/ratings.csv")
movies = pd.read_csv("dados/movies.csv")

dados = pd.merge(ratings, movies, on="movieId")

matriz = dados.pivot_table(
    index='userId',
    columns='title',
    values='rating'
).fillna(0)

similaridade = cosine_similarity(matriz)

similaridade_df = pd.DataFrame(
    similaridade,
    index=matriz.index,
    columns=matriz.index
)

# =========================
# FUNÇÃO DE RECOMENDAÇÃO
# =========================

def recomendar_filmes(usuario):

    usuarios_parecidos = similaridade_df[usuario].sort_values(
        ascending=False
    )

    usuarios_parecidos = usuarios_parecidos.drop(usuario)

    top_usuarios = usuarios_parecidos.head(10)

    filmes_assistidos = matriz.loc[usuario]
    filmes_assistidos = filmes_assistidos[
        filmes_assistidos > 0
    ].index.tolist()

    recomendacoes = {}

    for usuario_similar, score_similaridade in top_usuarios.items():

        filmes_usuario = matriz.loc[usuario_similar]
        filmes_usuario = filmes_usuario[filmes_usuario > 0]

        for filme, nota in filmes_usuario.items():

            if filme in filmes_assistidos:
                continue

            score = nota * score_similaridade

            if filme not in recomendacoes:
                recomendacoes[filme] = score
            else:
                recomendacoes[filme] += score

    recomendacoes_df = pd.DataFrame(
        recomendacoes.items(),
        columns=['Filme', 'Pontuacao']
    )

    recomendacoes_df = recomendacoes_df.sort_values(
        by='Pontuacao',
        ascending=False
    )

    return recomendacoes_df.head(10)['Filme'].tolist()

# =========================
# ENDPOINT DA API
# =========================

@app.route('/recommend/<int:usuario>', methods=['GET'])
def recommend(usuario):

    recomendacoes = recomendar_filmes(usuario)

    return jsonify({
        "usuario": usuario,
        "recomendacoes": recomendacoes
    })

# =========================
# INICIAR API
# =========================

if __name__ == '__main__':
    app.run(debug=True)