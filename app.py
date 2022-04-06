from flask import Flask, render_template, request, redirect, url_for
import reccommend

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/recommendation")
def recommendation():
    rec_type = request.args.get('type')
    recc_id = int(request.args.get('id'))
    reccommend.initialize()

    output = None
    if rec_type == 'recommend_to_users':
        output = reccommend.recommend_to_users([recc_id])

    elif rec_type == 'similar_users':
        output = reccommend.similar_users([recc_id])
    elif rec_type == 'similar_vns':
        output = reccommend.similar_vns([recc_id])

    if output is not None:
        output['type'] = rec_type
        return render_template('recommendation.html', output=output)
    else:
        return redirect(url_for('index.html'))



if __name__ == '__main__':
    app.run()

