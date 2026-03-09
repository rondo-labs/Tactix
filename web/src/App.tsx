import { Routes, Route } from "react-router-dom";
import ViewerPage from "./features/viewer/ViewerPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<ViewerPage />} />
    </Routes>
  );
}
