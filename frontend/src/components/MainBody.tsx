function MainBody() {
    return (
        <div className="flex-grow-1 p-4 bg-light">
            <h2 className="text-center">Upload Paper</h2>
            <div className="mt-3">
                <input type="file" className="form-control" accept=".pdf" />
            </div>
        </div>
    );
}

export default MainBody;
