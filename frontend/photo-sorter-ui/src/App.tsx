import { useState } from "react";
import axios from "axios";

type Clusters = {
  [key: string]: string[];
};

function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [clusters, setClusters] = useState<Clusters | null>(null);

  const handleUpload = async () => {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    setLoading(true);

    try {
      const response = await axios.post("http://13.239.35.7:8000/upload", formData);
      setClusters(response.data.clusters);
    } catch (err) {
      alert("Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mt-4">
      <h2>Trip Photo Face Sorter</h2>

      <input
        type="file"
        multiple
        className="form-control mb-3"
        onChange={(e) => e.target.files && setFiles(Array.from(e.target.files))}
      />

      <button className="btn btn-primary mb-4" onClick={handleUpload}>
        {loading ? "Processing..." : "Upload & Cluster"}
      </button>

      {clusters && Object.entries(clusters).map(([person, images]) => (
  <div key={person} className="card mb-4">
    <div className="card-header">
      <strong>{person}</strong>
    </div>
    <div
      className="card-body"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        const image = e.dataTransfer.getData("image");
        const fromPerson = e.dataTransfer.getData("fromPerson");

        if (fromPerson === person) return;

        setClusters((prev) => {
          if (!prev) return prev;

          const newClusters = { ...prev };

          newClusters[fromPerson] = newClusters[fromPerson].filter(
            (imgUrl) => imgUrl !== image
          );
          newClusters[person] = [...newClusters[person], image];

          return newClusters;
        });
      }}
    >
      <div className="row">
        {images.map((img, idx) => (
          <div className="col-md-3 mb-2" key={idx}>
            <img
              src={img}
              className="img-fluid rounded"
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData("image", img);
                e.dataTransfer.setData("fromPerson", person);
              }}
            />
          </div>
        ))}
      </div>
    </div>
  </div>
))}

    </div>
  );
}

export default App;
