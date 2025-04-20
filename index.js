import express from "express";
import multer from "multer";
import cors from "cors";

const app = express();
const upload = multer({ dest: "uploads/" });

app.use(cors());

app.get("/api/test", (req, res) => {
  res.send("API is live!");
});

app.post("/api/emotion-detect", upload.single("audio"), (req, res) => {
  if (!req.file) return res.status(400).send("No file uploaded.");
  console.log("Received file:", req.file.originalname);
  res.json({ emotion: "happy" }); // stub response
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log("Server started on port", PORT);
});
