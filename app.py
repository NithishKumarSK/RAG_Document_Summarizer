# app.py


from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from rag_engine import RAGEngine

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

rag = RAGEngine()

@app.route('/')
def index():
    """Serve the frontend"""
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "RAG API is running"}), 200

@app.route('/upload', methods=['POST'])
def upload_files():
    """Upload and process documents"""
    # DEBUG: Print what we received
    print("\n=== DEBUG INFO ===")
    print(f"Content-Type: {request.content_type}")
    print(f"request.files keys: {list(request.files.keys())}")
    print(f"request.form keys: {list(request.form.keys())}")
    print(f"request.data: {request.data[:100] if request.data else 'None'}")
    print("==================\n")
    
    # Try to get files with different key names
    files = None
    if 'files' in request.files:
        files = request.files.getlist('files')
        print(f"Found 'files' key with {len(files)} files")
    elif 'files[]' in request.files:
        files = request.files.getlist('files[]')
        print(f"Found 'files[]' key with {len(files)} files")
    else:
        available_keys = list(request.files.keys())
        print(f"Available keys in request.files: {available_keys}")
        
        if available_keys:
            # Use the first available key
            first_key = available_keys[0]
            files = request.files.getlist(first_key)
            print(f"Using key '{first_key}' with {len(files)} files")
        else:
            return jsonify({
                "error": "No files provided",
                "debug_info": {
                    "content_type": request.content_type,
                    "available_keys": available_keys,
                    "form_keys": list(request.form.keys())
                }
            }), 400
    
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400
    
    results = []
    
    for file in files:
        print(f"Processing file: {file.filename}")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                chunks = rag.add_document(filepath, filename)
                results.append({
                    "filename": filename,
                    "status": "success",
                    "chunks_created": chunks
                })
                print(f"✅ Success: {filename} - {chunks} chunks")
            except Exception as e:
                results.append({
                    "filename": filename,
                    "status": "failed",
                    "error": str(e)
                })
                print(f"❌ Error: {filename} - {str(e)}")
        else:
            results.append({
                "filename": file.filename if file else "unknown",
                "status": "failed",
                "error": "Invalid file type"
            })
            print(f"❌ Invalid file type: {file.filename if file else 'unknown'}")
    
    return jsonify({"results": results}), 200

@app.route('/query', methods=['POST'])
def query():
    """Query the document database"""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400
    
    question = data['question']
    n_results = data.get('n_results', 5)
    
    try:
        answer, sources = rag.query(question, n_results)
        
        if answer is None:
            return jsonify({
                "error": "No documents found. Please upload documents first."
            }), 404
        
        return jsonify({
            "question": question,
            "answer": answer,
            "sources": sources
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    """Generate summary of documents"""
    data = request.get_json() or {}
    filename = data.get('filename', None)
    
    try:
        summary = rag.summarize(filename)
        return jsonify({
            "summary": summary,
            "filename": filename if filename else "all documents"
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Get database statistics"""
    try:
        stats = rag.get_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear', methods=['DELETE'])
def clear():
    """Clear all documents from database"""
    try:
        rag.collection.delete()
        return jsonify({"message": "All documents cleared"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 RAG Document Summarizer API Starting...")
    print("📚 Endpoints available:")
    print("  - GET  /health       : Health check")
    print("  - POST /upload       : Upload documents")
    print("  - POST /query        : Ask questions")
    print("  - POST /summarize    : Get summary")
    print("  - GET  /stats        : Database stats")
    print("  - DELETE /clear      : Clear database")
    print("\n✅ Server running on http://localhost:5000")
    
    # app.run(host='0.0.0.0', port=5000, debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)