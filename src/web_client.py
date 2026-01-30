"""
Simple HTML client for testing the Spam Detection API
"""

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Detection API - Web Client</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            max-width: 600px;
            width: 100%;
            padding: 40px;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
            text-align: center;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
        }
        
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 14px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            width: 100%;
        }
        
        button:hover {
            transform: translateY(-2px);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button:disabled {
            opacity: 0.7;
            cursor: not-allowed;
        }
        
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 5px;
            display: none;
        }
        
        .result.show {
            display: block;
        }
        
        .result.spam {
            background: #ffebee;
            border-left: 4px solid #f44336;
        }
        
        .result.ham {
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
        }
        
        .result h3 {
            margin-bottom: 10px;
        }
        
        .result.spam h3 {
            color: #c62828;
        }
        
        .result.ham h3 {
            color: #2e7d32;
        }
        
        .result-details {
            display: flex;
            justify-content: space-between;
            margin-top: 15px;
            font-size: 14px;
        }
        
        .result-detail {
            flex: 1;
        }
        
        .result-label {
            color: #666;
            font-weight: 500;
        }
        
        .result-value {
            color: #333;
            margin-top: 5px;
            font-weight: 600;
        }
        
        .loading {
            display: none;
            text-align: center;
            color: #667eea;
            margin-top: 15px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
            display: none;
        }
        
        .error.show {
            display: block;
        }
        
        .status {
            text-align: center;
            margin-bottom: 20px;
            padding: 10px;
            border-radius: 5px;
            font-size: 14px;
        }
        
        .status.ok {
            background: #e8f5e9;
            color: #2e7d32;
        }
        
        .status.error {
            background: #ffebee;
            color: #c62828;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚨 Spam Detector</h1>
        <p class="subtitle">AI-powered SMS spam detection</p>
        
        <div id="status" class="status">
            <span id="statusText">Checking API connection...</span>
        </div>
        
        <form id="predictionForm">
            <div class="form-group">
                <label for="message">Enter a message:</label>
                <textarea id="message" name="message" rows="4" placeholder="Paste your message here..." required></textarea>
            </div>
            
            <button type="submit">Analyze Message</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing message...</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="result" id="result">
            <h3 id="resultTitle"></h3>
            <div class="result-details">
                <div class="result-detail">
                    <div class="result-label">Classification</div>
                    <div class="result-value" id="resultPrediction"></div>
                </div>
                <div class="result-detail">
                    <div class="result-label">Confidence</div>
                    <div class="result-value" id="resultConfidence"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const API_URL = 'http://localhost:8000';
        const predictionForm = document.getElementById('predictionForm');
        const messageInput = document.getElementById('message');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        const error = document.getElementById('error');
        const statusDiv = document.getElementById('status');
        
        // Check API status on page load
        async function checkAPIStatus() {
            try {
                const response = await fetch(`${API_URL}/health`);
                if (response.ok) {
                    const data = await response.json();
                    if (data.model_loaded) {
                        statusDiv.className = 'status ok';
                        statusDiv.innerHTML = '✓ API is ready. You can start testing!';
                    } else {
                        statusDiv.className = 'status error';
                        statusDiv.innerHTML = '⚠ Model not loaded. Run main_rf.py first.';
                    }
                } else {
                    throw new Error('API not responding');
                }
            } catch (err) {
                statusDiv.className = 'status error';
                statusDiv.innerHTML = '❌ Cannot connect to API. Make sure the server is running.';
            }
        }
        
        checkAPIStatus();
        
        // Handle form submission
        predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const message = messageInput.value.trim();
            if (!message) return;
            
            // Reset UI
            error.classList.remove('show');
            error.textContent = '';
            result.classList.remove('show');
            loading.style.display = 'block';
            
            try {
                const response = await fetch(`${API_URL}/predict`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message })
                });
                
                if (!response.ok) {
                    throw new Error(`API error: ${response.statusText}`);
                }
                
                const data = await response.json();
                displayResult(data);
            } catch (err) {
                displayError(err.message);
            } finally {
                loading.style.display = 'none';
            }
        });
        
        function displayResult(data) {
            const isSpam = data.is_spam;
            const resultTitle = document.getElementById('resultTitle');
            const resultPrediction = document.getElementById('resultPrediction');
            const resultConfidence = document.getElementById('resultConfidence');
            
            result.classList.add(isSpam ? 'spam' : 'ham');
            result.classList.remove(isSpam ? 'ham' : 'spam');
            
            resultTitle.textContent = isSpam ? '⚠️ SPAM DETECTED' : '✓ LEGITIMATE MESSAGE';
            resultPrediction.textContent = data.prediction.toUpperCase();
            resultConfidence.textContent = data.confidence;
            
            result.classList.add('show');
        }
        
        function displayError(message) {
            error.textContent = `Error: ${message}`;
            error.classList.add('show');
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    # Save the HTML to a file
    with open('client.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("✓ Web client saved to client.html")
    print("Open it in your browser to test the API")
