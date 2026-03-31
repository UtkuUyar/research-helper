import Sidebar from "./components/Sidebar";
import MainBody from "./components/MainBody";

function App() {
    return (
        <div className="d-flex vh-100">
            <Sidebar />
            <MainBody />
        </div>
    );
}

export default App;
