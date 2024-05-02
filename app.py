from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Mock newspaper snapshots data
newspaper_snapshots = {
    'NYTimes': 'https://www.livemint.com/lm-img/img/2024/04/29/600…engaluru_heatwave_1714357380982_1714357381172.jpg',
    'WashingtonPost': 'https://static.tnn.in/thumb/msid-109685969,thumbsi…width-1280,height-720,resizemode-75/109685969.jpg',
    'TheGuardian': 'https://s.w-x.co/util/image/w/in-kolkata%20rain_3.…p=16:9&width=980&format=pjpg&auto=webp&quality=60'

    
}


@app.route('/newspaper_snapshot', methods=['GET'])
def get_newspaper_snapshot():
    
    newspaper_name = request.args.get('newspaper')

    if not newspaper_name:
        return jsonify({'error': 'Newspaper parameter is required'}), 400

    
    if newspaper_name not in newspaper_snapshots:
        return jsonify({'error': 'Newspaper not found'}), 404

    
    snapshot_url = newspaper_snapshots[newspaper_name]
    return jsonify({'newspaper': newspaper_name, 'snapshot_url': snapshot_url})


if __name__ == '__main__':
    app.run(debug=True)
