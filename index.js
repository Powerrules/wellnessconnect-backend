const express = require("express");
const multer = require("multer");
const axios = require("axios");
const fs = require("fs");
const FormData = require("form-data");
const app = express();
const upload = multer({ dest: "uploads/" });

app.post("/api/emotion-detect", upload.single("audio"), async (req, res) => {
  const audioPath = req.file.path;

  try {
    const form = new FormData();
    form.append("audio", fs.createReadStream(audioPath));

    const response = await axios.post("http://127.0.0.1:5000", form, {
      headers: form.getHeaders(),
    });

    fs.unlinkSync(audioPath); // Cleanup uploaded file
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(10000, () => {
  console.log("Server started on port 10000");
});
