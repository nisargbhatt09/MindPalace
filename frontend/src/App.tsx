import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import HowToUse from "./pages/HowToUse";
import About from "./pages/About";
import Privacy from "./pages/Privacy";
import AppPage from "./pages/AppPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/how-to-use" element={<HowToUse />} />
        <Route path="/about" element={<About />} />
        <Route path="/privacy" element={<Privacy />} />
        <Route path="/app" element={<AppPage />} />
      </Route>
    </Routes>
  );
}
