import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [jd, setJd] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("job_desc", jd);

    const res = await fetch("http://localhost:8000/analyze", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    setResult(data);
  };

  return (
    <div>
      <h1>AI Resume Analyzer</h1>

      <textarea onChange={(e) => setJd(e.target.value)} />
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />

      <button onClick={handleSubmit}>Analyze</button>

      {result && (
        <div>
          <h2>Score: {result.score}</h2>
          <pre>{result.analysis}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
